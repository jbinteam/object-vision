"""Reuse a target search tree and correspondences within one ICP registration.

The numerical update is still Open3D's point-to-point/point-to-plane estimator.
Its correction acts on a temporary transformed source: T_new = delta @ T_old.
Only the absolute object-to-camera transform leaves this adapter.
"""
from dataclasses import dataclass
import numpy as np


def build_search_index(o3d, points):
    # Float64 keeps correspondence distances at legacy Open3D precision. The
    # tensor API batches native searches instead of crossing Python per point.
    search = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor.from_numpy(points))
    search.knn_index()
    return search


@dataclass
class ICPStep:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_set: np.ndarray
    failure_reason: str = ''


class CachedICPWorkspace:
    def __init__(self, o3d, reference_points, target, estimator, maximum_distance, initial):
        self.o3d, self.reference, self.target = o3d, reference_points, target
        self.estimator, self.maximum_distance = estimator, maximum_distance
        self.search = build_search_index(o3d, np.asarray(target.points))
        self.source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference_points.copy()))
        self.points = np.asarray(self.source.points)
        self.query = o3d.core.Tensor.from_numpy(self.points)
        self.transform = np.array(initial, dtype=np.float64, copy=True)
        self.points[:] = self.reference @ self.transform[:3, :3].T + self.transform[:3, 3]
        self.pairs, self.fitness, self.rmse = self.evaluate()

    def evaluate(self):
        indices, squared_distances = self.search.knn_search(self.query, 1)
        nearest = indices.numpy().ravel()
        distance_squared = squared_distances.numpy().ravel()
        valid = distance_squared < self.maximum_distance ** 2
        source_indices = np.flatnonzero(valid)
        pairs = np.column_stack((source_indices, nearest[valid])).astype(np.int32)
        count = len(pairs)
        rmse = float(np.sqrt(np.mean(distance_squared[valid]))) if count else 0.0
        return pairs, count / len(self.reference), rmse

    def step(self):
        update = np.asarray(self.estimator.compute_transformation(
            self.source, self.target, self.o3d.utility.Vector2iVector(self.pairs)))
        candidate = update @ self.transform
        if candidate.shape != (4, 4) or not np.isfinite(candidate).all():
            return ICPStep(candidate, float('nan'), float('nan'), np.empty((0, 2), np.int32))
        self.transform = candidate
        # Rebuild from immutable object coordinates, never from an observed cloud.
        self.points[:] = self.reference @ candidate[:3, :3].T + candidate[:3, 3]
        self.pairs, self.fitness, self.rmse = self.evaluate()
        return ICPStep(candidate.copy(), self.fitness, self.rmse, self.pairs)
