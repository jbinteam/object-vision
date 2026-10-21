"""Robust keypoint PnP for the canonical object frame.

Transform convention (used by PnP, GICP, TF, and the reference cloud):

    P_camera = T_camera_object @ P_object

The returned transform is therefore never inverted before it is used as the
reference-cloud registration seed.
"""
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class PnPConfig:
    keypoint_confidence_threshold: float = 0.35
    min_valid_keypoints: int = 4
    pnp_ransac_iterations: int = 200
    pnp_reprojection_threshold: float = 5.0
    pnp_confidence: float = 0.99
    min_pnp_inliers: int = 4
    max_mean_reprojection_error: float = 5.0
    depth_validation_enabled: bool = True
    max_depth_translation_error: float = 0.12

    def __post_init__(self):
        if not 0 <= self.keypoint_confidence_threshold <= 1:
            raise ValueError('keypoint_confidence_threshold must be in [0,1]')
        if self.min_valid_keypoints < 4 or self.min_pnp_inliers < 4:
            raise ValueError('PnP needs at least four valid keypoints/inliers')
        if self.pnp_ransac_iterations < 1:
            raise ValueError('pnp_ransac_iterations must be positive')
        if self.pnp_reprojection_threshold <= 0 or self.max_mean_reprojection_error <= 0:
            raise ValueError('Reprojection thresholds must be positive')
        if not 0 < self.pnp_confidence < 1:
            raise ValueError('pnp_confidence must be in (0,1)')
        if self.max_depth_translation_error <= 0:
            raise ValueError('max_depth_translation_error must be positive')


@dataclass
class PnPResult:
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    accepted: bool = False
    reason: str = 'not_run'
    valid_keypoints: int = 0
    inlier_count: int = 0
    inlier_ids: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    projected_keypoints: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    reprojection_errors: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_reprojection_error: float = float('inf')
    median_reprojection_error: float = float('inf')
    max_reprojection_error: float = float('inf')
    depth_translation_error: float = float('nan')
    pnp_ms: float = 0.0


def load_object_keypoints(path):
    """Load ordered object-frame keypoints; IDs must be contiguous from zero."""
    import yaml
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'object_keypoints_path does not exist: {path}')
    data = yaml.safe_load(path.read_text())
    if data.get('units') != 'm':
        raise ValueError('Object keypoints must use metre units')
    entries = data.get('object_keypoints', [])
    ids = [entry.get('id') for entry in entries]
    if ids != list(range(len(entries))) or len(entries) < 4:
        raise ValueError('Keypoint IDs must be unique, ordered, and contiguous from zero')
    points = np.asarray([entry.get('xyz') for entry in entries], dtype=np.float64)
    if points.shape != (len(entries), 3) or not np.isfinite(points).all():
        raise ValueError('Every object keypoint needs finite xyz coordinates')
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-8) < 2:
        raise ValueError('Object keypoints are degenerate')
    return points, data


def camera_matrix(values):
    matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.isclose(matrix[2, 2], 1)):
        raise ValueError('Invalid camera intrinsic matrix')
    return matrix


def project_keypoints(object_points, transform, intrinsic, distortion=None):
    import cv2
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError('Transform must be 4x4')
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    distortion = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
    projected, _ = cv2.projectPoints(np.asarray(object_points, dtype=np.float64), rvec,
                                     transform[:3, 3], camera_matrix(intrinsic), distortion)
    return projected.reshape(-1, 2)


def depth_corrected_seed(transform, observed_centroid, trigger=0.02, maximum=0.08):
    """Optionally correct monocular PnP's Z scale before local registration.

    The PnP result itself remains unchanged and is still published/debugged.
    Only the GICP seed is shifted, and only when the masked depth median exposes
    a material scale error. X/Y and orientation remain purely PnP-derived.
    """
    seed = np.asarray(transform, dtype=np.float64).copy()
    centroid = np.asarray(observed_centroid, dtype=np.float64)
    if seed.shape != (4, 4) or centroid.shape != (3,):
        raise ValueError('Depth seed correction needs a 4x4 pose and XYZ centroid')
    if not np.isfinite(seed).all() or not np.isfinite(centroid).all():
        raise ValueError('Depth seed correction inputs must be finite')
    if not 0 <= trigger < maximum:
        raise ValueError('Require 0 <= depth correction trigger < maximum')
    correction = float(centroid[2] - seed[2, 3])
    applied = abs(correction) >= trigger and abs(correction) <= maximum
    if applied:
        seed[2, 3] += correction
    return seed, correction, applied


class RobustPnP:
    """EPNP-RANSAC initializer followed by iterative inlier refinement.

    EPNP is used for the non-coplanar cuboid keypoints because it accepts any
    count >= 4. RANSAC rejects learned-keypoint outliers. The consensus pose is
    then refined with the iterative method for lower reprojection error.
    """
    def __init__(self, object_points, config=PnPConfig()):
        points = np.asarray(object_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
            raise ValueError('object_points must be finite N x 3 with N >= 4')
        self.object_points = points.copy()
        self.config = config

    def estimate(self, image_keypoints, intrinsic, distortion=None, observed_centroid=None):
        import cv2
        started = time.perf_counter()
        result = PnPResult(projected_keypoints=np.full((len(self.object_points), 2), np.nan))
        detected = np.asarray(image_keypoints, dtype=np.float64)
        if detected.ndim != 2 or detected.shape[0] != len(self.object_points) or detected.shape[1] not in (3, 4):
            result.reason = 'keypoint_count_or_shape_mismatch'
            return result
        valid = (np.isfinite(detected[:, :3]).all(axis=1)
                 & (detected[:, 2] >= self.config.keypoint_confidence_threshold))
        if detected.shape[1] == 4:
            valid &= detected[:, 3].astype(bool)
        ids = np.flatnonzero(valid)
        result.valid_keypoints = len(ids)
        if len(ids) < self.config.min_valid_keypoints:
            result.reason = 'insufficient_visible_keypoints'
            return result
        k = camera_matrix(intrinsic)
        dist = np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            self.object_points[ids], detected[ids, :2], k, dist,
            iterationsCount=self.config.pnp_ransac_iterations,
            reprojectionError=self.config.pnp_reprojection_threshold,
            confidence=self.config.pnp_confidence, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None:
            result.reason = 'pnp_ransac_failed'
            return result
        local_inliers = inliers.reshape(-1)
        result.inlier_ids = ids[local_inliers].astype(np.int32)
        result.inlier_count = len(result.inlier_ids)
        if result.inlier_count < self.config.min_pnp_inliers:
            result.reason = 'insufficient_pnp_inliers'
            return result
        # Refine only the RANSAC consensus. useExtrinsicGuess retains its basin.
        success, rvec, tvec = cv2.solvePnP(
            self.object_points[result.inlier_ids], detected[result.inlier_ids, :2], k, dist,
            rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            result.reason = 'pnp_refinement_failed'
            return result
        rotation, _ = cv2.Rodrigues(rvec)
        result.transform[:3, :3] = rotation
        result.transform[:3, 3] = tvec.reshape(3)
        if result.transform[2, 3] <= 0 or not np.isfinite(result.transform).all():
            result.reason = 'pnp_pose_behind_camera_or_nonfinite'
            return result
        result.projected_keypoints = project_keypoints(self.object_points, result.transform, k, dist)
        result.reprojection_errors = np.linalg.norm(
            result.projected_keypoints[result.inlier_ids] - detected[result.inlier_ids, :2], axis=1)
        result.mean_reprojection_error = float(np.mean(result.reprojection_errors))
        result.median_reprojection_error = float(np.median(result.reprojection_errors))
        result.max_reprojection_error = float(np.max(result.reprojection_errors))
        if result.mean_reprojection_error > self.config.max_mean_reprojection_error:
            result.reason = 'high_mean_reprojection_error'
            return result
        if self.config.depth_validation_enabled and observed_centroid is not None:
            centroid = np.asarray(observed_centroid, dtype=np.float64)
            if centroid.shape != (3,) or not np.isfinite(centroid).all():
                result.reason = 'invalid_observed_depth_centroid'
                return result
            result.depth_translation_error = float(np.linalg.norm(result.transform[:3, 3] - centroid))
            if result.depth_translation_error > self.config.max_depth_translation_error:
                result.reason = 'pnp_depth_inconsistency'
                return result
        result.accepted, result.reason = True, 'accepted'
        result.pnp_ms = (time.perf_counter() - started) * 1000
        return result
