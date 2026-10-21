"""Fixed-reference ICP. T_camera_object maps object column vectors to camera.

The caller supplies a previous accepted pose in Phase 3, or uses independent
centroid initialization in Phase 2. The absolute transform is returned
directly, never multiplied by the seed a second time.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .icp_workspace import CachedICPWorkspace


@dataclass(frozen=True)
class ICPConfig:
    voxel_size: float = 0.003
    minimum_points: int = 30
    icp_type: str = 'point_to_plane'
    icp_backend: str = 'cached'
    icp_cuda_device: int = 0
    icp_max_iterations: int = 60
    icp_max_correspondence_distance: float = 0.025
    icp_transformation_epsilon: float = 1e-5
    icp_rotation_epsilon: float = 1e-4
    icp_time_budget: float = 0.25
    normal_radius: float = 0.015
    normal_max_neighbors: int = 30
    fitness_threshold: float = 0.35
    rmse_threshold: float = 0.012
    minimum_correspondences: int = 30

    def __post_init__(self):
        if self.icp_type not in ('point_to_point', 'point_to_plane'):
            raise ValueError('icp_type must be point_to_point or point_to_plane')
        if self.icp_backend not in ('cached', 'open3d', 'cuda', 'cuda_legacy'):
            raise ValueError('icp_backend must be cached, open3d, cuda, or cuda_legacy')
        if not isinstance(self.icp_cuda_device, int) or self.icp_cuda_device < 0:
            raise ValueError('icp_cuda_device must be a nonnegative integer')
        for name in ('icp_max_correspondence_distance', 'icp_transformation_epsilon',
                     'icp_rotation_epsilon', 'icp_time_budget', 'normal_radius', 'rmse_threshold'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.voxel_size) or self.voxel_size < 0:
            raise ValueError('voxel_size must be nonnegative')
        if not 0 < self.fitness_threshold <= 1:
            raise ValueError('fitness_threshold must be in (0,1]')
        if self.minimum_points < 3 or self.minimum_correspondences < 3 or self.icp_max_iterations < 1 or self.normal_max_neighbors < 3:
            raise ValueError('Invalid point/iteration/neighbor counts')


def is_rigid(transform):
    t = np.asarray(transform)
    return (t.shape == (4, 4) and np.isfinite(t).all() and np.allclose(t[3], [0, 0, 0, 1], atol=1e-7)
            and np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5))


def motion_delta(previous, current):
    delta = np.linalg.inv(previous) @ current
    return float(np.linalg.norm(delta[:3, 3])), float(Rotation.from_matrix(delta[:3, :3]).magnitude())


def usable_geometry(points, minimum_points):
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_points or not np.isfinite(points).all():
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    # Three-dimensional or planar surfaces are valid; lines and repeated points are not.
    return singular[0] > 1e-8 and singular[1] > singular[0] * 1e-3


def centroid_initial_guess(reference, observed, rotation=None):
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    transform[:3, 3] = observed.mean(axis=0) - transform[:3, :3] @ reference.mean(axis=0)
    if not is_rigid(transform):
        raise ValueError('Invalid initial rotation')
    return transform


def initial_rotation(reference_path, mode='identity', degrees=(0., 0., 0.)):
    if mode == 'identity':
        return np.eye(3)
    if mode == 'manual':
        angles = np.asarray(degrees)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError('initial_rotation_deg must contain finite roll, pitch, yaw')
        return Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    if mode == 'reference_metadata':
        metadata = json.loads(Path(reference_path).with_name('frame.json').read_text())
        transform = np.asarray(metadata['T_scan_obj'], dtype=float)
        if metadata.get('units') != 'm' or not is_rigid(transform):
            raise ValueError('Reference metadata must contain a rigid T_scan_obj in metres')
        # Orientation hint only. Translation is recomputed from the live centroid.
        return transform[:3, :3].copy()
    raise ValueError('initial_rotation_mode must be identity, manual, or reference_metadata')


@dataclass
class RegistrationResult:
    transform: np.ndarray
    accepted: bool = False
    converged: bool = False
    reason: str = 'not_run'
    fitness: float = 0.0
    rmse: float = float('inf')
    correspondences: int = 0
    iterations: int = 0
    icp_ms: float = 0.0
    preparation_ms: float = 0.0


class FixedReferenceICP:
    def __init__(self, reference_path, config=ICPConfig()):
        import open3d as o3d
        self.o3d, self.config = o3d, config
        path = Path(reference_path)
        if not path.is_file() or path.suffix.lower() not in ('.ply', '.pcd'):
            raise ValueError('reference_cloud_path must be an existing .ply or .pcd file')
        cloud = o3d.io.read_point_cloud(str(path))
        if not usable_geometry(np.asarray(cloud.points), config.minimum_points):
            raise ValueError('Reference has too few, nonfinite, or degenerate points')
        if config.voxel_size:
            cloud = cloud.voxel_down_sample(config.voxel_size)
        if not usable_geometry(np.asarray(cloud.points), config.minimum_points):
            raise ValueError('Reference is too small after downsampling')
        self._reference = cloud  # load/preprocess once; never transform this object
        self.reference_points = np.asarray(cloud.points).copy()
        self.reference_points.setflags(write=False)
        self.centroid = self.reference_points.mean(axis=0)
        self.cuda_reference = None
        self.cuda_plane = None
        if config.icp_backend in ('cuda', 'cuda_legacy'):
            from .cuda_icp import cuda_device
            device = cuda_device(o3d, config.icp_cuda_device)
            self.cuda_reference = o3d.core.Tensor(self.reference_points.copy(), o3d.core.float64, device)
            if config.icp_backend == 'cuda' and config.icp_type == 'point_to_plane':
                from .cuda_plane import CudaPlaneEngine
                self.cuda_plane = CudaPlaneEngine(o3d, self.cuda_reference)

    def register(self, observed, rotation=None, initial_guess=None):
        cfg, o3d = self.config, self.o3d
        started = time.perf_counter()
        result = RegistrationResult(np.eye(4))
        observed = np.asarray(observed, dtype=np.float64)
        if not usable_geometry(observed, cfg.minimum_points):
            result.reason = 'invalid_observed_cloud'
            return result
        target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
        if cfg.voxel_size:
            target = target.voxel_down_sample(cfg.voxel_size)
        points = np.asarray(target.points)
        if not usable_geometry(points, cfg.minimum_points):
            result.reason = 'insufficient_preprocessed_points'
            return result
        if cfg.icp_type == 'point_to_plane':
            target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=cfg.normal_radius, max_nn=cfg.normal_max_neighbors))
            normals = np.asarray(target.normals)
            if not np.isfinite(normals).all() or (np.linalg.norm(normals, axis=1) < .5).any():
                result.reason = 'invalid_target_normals'
                return result
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        transform = (centroid_initial_guess(self.reference_points, points, rotation)
                     if initial_guess is None else np.asarray(initial_guess, dtype=float).copy())
        if not is_rigid(transform):
            result.reason = 'invalid_initial_guess'
            return result
        result.preparation_ms = (time.perf_counter() - started) * 1000
        if self.cuda_plane is not None and len(points) <= self.cuda_plane.capacity:
            return self.cuda_plane.register(target, transform, cfg, result, started)
        icp_start = time.perf_counter()
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        workspace = (CachedICPWorkspace(o3d, self.reference_points, target, estimator,
                                        cfg.icp_max_correspondence_distance, transform)
                     if cfg.icp_backend == 'cached' else None)
        if cfg.icp_backend in ('cuda', 'cuda_legacy'):
            from .cuda_icp import TensorICPWorkspace
            workspace = TensorICPWorkspace(o3d, self.cuda_reference, target, cfg.icp_type,
                                           cfg.icp_max_correspondence_distance, transform)
        result.reason = 'iteration_limit'
        for iteration in range(cfg.icp_max_iterations):
            if time.perf_counter() - started > cfg.icp_time_budget:
                result.reason = 'time_budget_exceeded'
                break
            step = (workspace.step() if workspace is not None else
                    o3d.pipelines.registration.registration_icp(
                        self._reference, target, cfg.icp_max_correspondence_distance,
                        transform, estimator, criteria))
            candidate = np.asarray(step.transformation).copy()
            result.iterations = iteration + 1
            result.fitness, result.rmse = float(step.fitness), float(step.inlier_rmse)
            pairs = np.asarray(step.correspondence_set)
            result.correspondences = len(pairs)
            if getattr(step, 'failure_reason', ''):
                result.reason = step.failure_reason
                break
            if not is_rigid(candidate) or not np.isfinite([result.fitness, result.rmse]).all():
                result.reason = 'invalid_registration_result'
                break
            result.transform = candidate
            if len(pairs) < cfg.minimum_correspondences:
                result.reason = 'insufficient_correspondences'
                break
            if cfg.icp_type == 'point_to_plane':
                # Reject underconstrained plane fits (a zero update is not convergence).
                source = self.reference_points[pairs[:, 0]] @ candidate[:3, :3].T + candidate[:3, 3]
                normal = np.asarray(target.normals)[pairs[:, 1]]
                centered = source - source.mean(axis=0)
                jacobian = np.column_stack((np.cross(centered, normal), normal))
                singular = np.linalg.svd(jacobian, compute_uv=False)
                if singular[-1] <= singular[0] * 1e-6:
                    result.reason = 'underconstrained_point_to_plane'
                    break
            translation, angle = motion_delta(transform, candidate)
            transform = candidate
            if translation <= cfg.icp_transformation_epsilon and angle <= cfg.icp_rotation_epsilon:
                result.converged = True
                if result.fitness < cfg.fitness_threshold:
                    result.reason = 'low_fitness'
                elif result.rmse > cfg.rmse_threshold:
                    result.reason = 'high_rmse'
                else:
                    result.accepted, result.reason = True, 'accepted'
                break
        result.icp_ms = (time.perf_counter() - icp_start) * 1000
        if time.perf_counter() - started > cfg.icp_time_budget:
            result.accepted, result.reason = False, 'time_budget_exceeded'
        return result
