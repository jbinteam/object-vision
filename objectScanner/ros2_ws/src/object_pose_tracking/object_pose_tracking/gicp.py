"""Coarse-to-fine GICP refinement of T_camera_object.

The reference is permanently stored in ``object_frame``. Open3D receives the
absolute PnP/previous-pose transform as its initializer, so its result directly
maps object coordinates into camera coordinates:

    P_camera = T_camera_object @ P_object
"""
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np

from .registration import is_rigid, usable_geometry


@dataclass(frozen=True)
class GICPLevel:
    name: str
    voxel_size: float
    max_correspondence_distance: float
    max_iterations: int
    transformation_epsilon: float = 1e-6
    enabled: bool = True

    def __post_init__(self):
        if not self.name or self.voxel_size < 0 or self.max_correspondence_distance <= 0:
            raise ValueError('Invalid GICP level geometry')
        if self.max_iterations < 1 or self.transformation_epsilon <= 0:
            raise ValueError('Invalid GICP convergence settings')


def default_levels():
    return (GICPLevel('coarse', .010, .035, 25, 1e-5),
            GICPLevel('medium', .005, .020, 20, 3e-6),
            GICPLevel('fine', .002, .010, 15, 1e-6))


@dataclass(frozen=True)
class GICPConfig:
    levels: tuple = field(default_factory=default_levels)
    minimum_points: int = 30
    minimum_correspondences: int = 30
    fitness_threshold: float = 0.20
    rmse_threshold: float = 0.012
    overlap_threshold: float = 0.15
    visibility_filter_enabled: bool = True
    visibility_normal_facing: bool = True
    tracking_fine_only: bool = True

    def __post_init__(self):
        if not self.levels or not any(level.enabled for level in self.levels):
            raise ValueError('At least one GICP level must be enabled')
        if self.minimum_points < 3 or self.minimum_correspondences < 3:
            raise ValueError('GICP point counts must be >= 3')
        for name in ('fitness_threshold', 'overlap_threshold'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if self.rmse_threshold <= 0:
            raise ValueError('rmse_threshold must be positive')


@dataclass
class GICPResult:
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    accepted: bool = False
    converged: bool = False
    reason: str = 'not_run'
    fitness: float = 0.0
    rmse: float = float('inf')
    overlap: float = 0.0
    correspondences: int = 0
    visible_reference_points: int = 0
    level_ms: dict = field(default_factory=dict)
    total_ms: float = 0.0


class ReferenceVisibilityFilter:
    """Approximate visible surface using depth, FOV, and outward normals.

    This deliberately does not claim to solve self-occlusion. Its interface is
    isolated so a z-buffer implementation can replace it without changing GICP.
    """
    def __init__(self, normal_facing=True):
        self.normal_facing = bool(normal_facing)

    def filter(self, points, transform, intrinsic=None, image_size=None, normals=None):
        points = np.asarray(points, dtype=np.float64)
        camera = points @ transform[:3, :3].T + transform[:3, 3]
        keep = camera[:, 2] > 1e-5
        if intrinsic is not None and image_size is not None:
            k = np.asarray(intrinsic, dtype=float).reshape(3, 3)
            height, width = image_size
            u = k[0, 0] * camera[:, 0] / camera[:, 2] + k[0, 2]
            v = k[1, 1] * camera[:, 1] / camera[:, 2] + k[1, 2]
            keep &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if self.normal_facing and normals is not None and len(normals) == len(points):
            normal_camera = np.asarray(normals) @ transform[:3, :3].T
            # Outward normal points toward the camera when dot(n, camera->surface) < 0.
            keep &= np.einsum('ij,ij->i', normal_camera, camera) < 0
        return np.flatnonzero(keep)


class CoarseToFineGICP:
    def __init__(self, reference_path, config=GICPConfig()):
        import open3d as o3d
        path = Path(reference_path)
        if not path.is_file() or path.suffix.lower() not in ('.ply', '.pcd'):
            raise ValueError('reference_cloud_path must be an existing .ply or .pcd file')
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
        if not usable_geometry(points, config.minimum_points):
            raise ValueError('Reference has insufficient or degenerate geometry')
        self.o3d, self.config = o3d, config
        self.reference_points = points.copy()
        self.reference_points.setflags(write=False)
        normals = np.asarray(cloud.normals, dtype=np.float64)
        self.reference_normals = normals.copy() if normals.shape == points.shape else None
        self.visibility = ReferenceVisibilityFilter(config.visibility_normal_facing)
        self._full_levels = {level.name: self._downsample(self.reference_points, level.voxel_size)
                             for level in config.levels if level.enabled}

    def _cloud(self, points):
        return self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(
            np.array(points, dtype=np.float64, order='C', copy=True)))

    def _downsample(self, points, voxel):
        cloud = self._cloud(points)
        return cloud.voxel_down_sample(voxel) if voxel else cloud

    def register(self, observed, initial_guess, intrinsic=None, image_size=None, tracking=False):
        cfg, o3d = self.config, self.o3d
        started = time.perf_counter()
        result = GICPResult(transform=np.asarray(initial_guess, dtype=float).copy())
        if not is_rigid(result.transform):
            result.reason = 'invalid_initial_guess'
            return result
        observed = np.asarray(observed, dtype=np.float64)
        if not usable_geometry(observed, cfg.minimum_points):
            result.reason = 'invalid_observed_cloud'
            return result
        reference_ids = np.arange(len(self.reference_points))
        if cfg.visibility_filter_enabled:
            reference_ids = self.visibility.filter(
                self.reference_points, result.transform, intrinsic, image_size,
                self.reference_normals)
        result.visible_reference_points = len(reference_ids)
        if len(reference_ids) < cfg.minimum_points:
            result.reason = 'insufficient_visible_reference_points'
            return result
        visible = self.reference_points[reference_ids]
        final_source_count = final_target_count = 0
        for level in cfg.levels:
            if not level.enabled:
                continue
            if tracking and cfg.tracking_fine_only and level.name != 'fine':
                continue
            level_start = time.perf_counter()
            source = (self._downsample(visible, level.voxel_size)
                      if cfg.visibility_filter_enabled else self._full_levels[level.name])
            target = self._downsample(observed, level.voxel_size)
            final_source_count, final_target_count = len(source.points), len(target.points)
            if min(final_source_count, final_target_count) < cfg.minimum_points:
                result.reason = f'insufficient_{level.name}_points'
                return result
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=level.transformation_epsilon,
                relative_rmse=level.transformation_epsilon,
                max_iteration=level.max_iterations)
            step = o3d.pipelines.registration.registration_generalized_icp(
                source, target, level.max_correspondence_distance, result.transform,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(), criteria)
            candidate = np.asarray(step.transformation).copy()
            if not is_rigid(candidate) or not np.isfinite([step.fitness, step.inlier_rmse]).all():
                result.reason = f'invalid_{level.name}_result'
                return result
            result.transform = candidate
            result.fitness = float(step.fitness)
            result.rmse = float(step.inlier_rmse)
            pairs = np.asarray(step.correspondence_set)
            result.correspondences = len(pairs)
            # Target coverage is more informative than source fitness for a
            # partial(observed)-to-full(reference) fit. Count each observed
            # target point once even if several reference points select it.
            result.overlap = (len(np.unique(pairs[:, 1])) / final_target_count
                              if len(pairs) and final_target_count else 0.)
            result.level_ms[level.name] = (time.perf_counter() - level_start) * 1000
        if not result.level_ms:
            result.reason = 'no_enabled_tracking_level'
            return result
        result.converged = True
        if result.correspondences < cfg.minimum_correspondences:
            result.reason = 'insufficient_correspondences'
        elif result.fitness < cfg.fitness_threshold:
            result.reason = 'low_fitness'
        elif result.rmse > cfg.rmse_threshold:
            result.reason = 'high_rmse'
        elif result.overlap < cfg.overlap_threshold:
            result.reason = 'low_overlap'
        else:
            result.accepted, result.reason = True, 'accepted'
        result.total_ms = (time.perf_counter() - started) * 1000
        return result
