"""Pure geometry in metres. T_A_B maps column vectors in B into A."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CloudConfig:
    minimum_depth: float = 0.15
    maximum_depth: float = 1.5
    voxel_size: float = 0.003
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    minimum_points: int = 30

    def __post_init__(self):
        if not (0 < self.minimum_depth < self.maximum_depth < float('inf')):
            raise ValueError('Require 0 < minimum_depth < maximum_depth')
        if not np.isfinite(self.voxel_size) or self.voxel_size < 0:
            raise ValueError('voxel_size must be finite and nonnegative')
        if self.outlier_neighbors < 0 or self.minimum_points < 3:
            raise ValueError('Invalid neighbor count or minimum_points')
        if not np.isfinite(self.outlier_std_ratio) or self.outlier_std_ratio <= 0:
            raise ValueError('outlier_std_ratio must be positive')


def backproject(depth, mask, intrinsic, config, encoding='16UC1', rgb=None):
    """Optical XYZ; optionally return matching uint8 RGB from the same pixels."""
    depth, mask, k = np.asarray(depth), np.asarray(mask), np.asarray(intrinsic)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError('Mask and aligned depth must have identical H x W shapes')
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError('Expected standard pinhole intrinsics')
    if encoding == '16UC1' and depth.dtype == np.uint16:
        z = depth.astype(np.float32) * 0.001
    elif encoding == '32FC1' and depth.dtype == np.float32:
        z = depth
    else:
        raise ValueError('Depth must be 16UC1 millimetres or 32FC1 metres')
    valid = (mask != 0) & np.isfinite(z) & (z >= config.minimum_depth) & (z <= config.maximum_depth)
    v, u = np.nonzero(valid)
    z = z[v, u]
    points = np.column_stack(((u - k[0, 2]) * z / k[0, 0],
                              (v - k[1, 2]) * z / k[1, 1], z)).astype(np.float32)
    if rgb is None:
        return points
    rgb = np.asarray(rgb)
    if rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError('RGB must be aligned uint8 H x W x 3 matching depth')
    return points, np.ascontiguousarray(rgb[v, u])


def _voxel_groups(keys):
    """Group XYZ integer keys in the same lexicographic order as unique(axis=0)."""
    lower, upper = keys.min(axis=0), keys.max(axis=0)
    # Python integers prevent the range/product check itself from overflowing.
    nx, ny, nz = [int(hi) - int(lo) + 1 for lo, hi in zip(lower, upper)]
    if nx * ny * nz > np.iinfo(np.int64).max:
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    else:
        # This monotone mixed-radix code retains sorted XYZ order without collisions.
        codes = ((keys[:, 0] - lower[0]) * ny + (keys[:, 1] - lower[1])) * nz + (keys[:, 2] - lower[2])
        _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    return inverse, counts


def preprocess(points, config, colors=None):
    """Filter points and optional paired RGB; average colors within each voxel."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != points.shape or colors.dtype != np.uint8:
            raise ValueError('Colors must be uint8 N x 3 matching the points')
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    if config.voxel_size > 0 and len(points):
        keys = np.floor(points / config.voxel_size).astype(np.int64)
        inverse, counts = _voxel_groups(keys)
        sums = np.column_stack([
            np.bincount(inverse, weights=points[:, axis], minlength=len(counts))
            for axis in range(3)])
        points = (sums / counts[:, None]).astype(np.float32)
        if colors is not None:
            color_sums = np.column_stack([
                np.bincount(inverse, weights=colors[:, axis], minlength=len(counts))
                for axis in range(3)])
            colors = np.rint(color_sums / counts[:, None]).clip(0, 255).astype(np.uint8)
    if config.outlier_neighbors > 0 and len(points) > 2:
        k = min(config.outlier_neighbors + 1, len(points))
        distances, _ = cKDTree(points).query(points, k=k, workers=1)
        means = distances[:, 1:].mean(axis=1)
        keep = means <= means.mean() + config.outlier_std_ratio * means.std()
        points = points[keep]
        if colors is not None:
            colors = colors[keep]
    points = np.ascontiguousarray(points)
    return points if colors is None else (points, np.ascontiguousarray(colors))


def transform_points(points, translation, quaternion_xyzw):
    """Apply a calibrated TF, never just relabel optical points as camera_link."""
    t, q = np.asarray(translation), np.asarray(quaternion_xyzw)
    if t.shape != (3,) or q.shape != (4,) or not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError('Invalid TF')
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-3):
        raise ValueError('TF quaternion is not normalized')
    return (np.asarray(points) @ Rotation.from_quat(q).as_matrix().T + t).astype(np.float32)
