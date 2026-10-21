"""Compatibility CUDA loop using Open3D tensor estimators.

Used by cuda_legacy, and by cuda for point-to-point / oversized target clouds.
The default CUDA point-to-plane loop lives in cuda_plane.py.

The immutable reference stays on the GPU across observations. Target points and
normals upload once per fit. Only the small transform and correspondence metrics
return to the CPU for the shared quality/convergence checks and ROS output.
"""
import numpy as np
from .icp_workspace import ICPStep


def cuda_device(o3d, index):
    if not o3d.core.cuda.is_available() or index >= o3d.core.cuda.device_count():
        raise RuntimeError(f'Open3D CUDA:{index} is unavailable; use the CUDA environment '
                           'or explicitly select icp_backend=cached for CPU')
    return o3d.core.Device(f'CUDA:{index}')


class TensorICPWorkspace:
    def __init__(self, o3d, reference, target, method, maximum_distance, initial):
        self.o3d, self.reference = o3d, reference
        self.device, self.dtype = reference.device, reference.dtype
        self.maximum_distance = maximum_distance
        self.target = o3d.t.geometry.PointCloud(self.tensor(np.asarray(target.points)))
        registration = o3d.t.pipelines.registration
        if method == 'point_to_plane':
            self.target.point.normals = self.tensor(np.asarray(target.normals))
            self.estimator = registration.TransformationEstimationPointToPlane()
        else:
            self.estimator = registration.TransformationEstimationPointToPoint()
        self.search = o3d.core.nns.NearestNeighborSearch(self.target.point.positions)
        self.search.hybrid_index(maximum_distance)
        self.transform = np.array(initial, dtype=np.float64, copy=True)
        self.source = o3d.t.geometry.PointCloud(self.device)
        self.update_source()
        self.evaluate()

    def tensor(self, values):
        return self.o3d.core.Tensor(np.asarray(values), self.dtype, self.device)

    def update_source(self):
        # Recompute from object coordinates; never overwrite the cached reference.
        rotation = self.tensor(self.transform[:3, :3].T.copy())
        translation = self.tensor(self.transform[:3, 3].copy())
        self.source.point.positions = self.reference.matmul(rotation) + translation

    def evaluate(self):
        indices, distances, _ = self.search.hybrid_search(
            self.source.point.positions, self.maximum_distance, 1)
        self.correspondences = indices.reshape((-1,))
        nearest = self.correspondences.cpu().numpy()
        squared = distances.cpu().numpy().ravel()
        valid = (nearest >= 0) & (squared < self.maximum_distance ** 2)
        # Keep the tensor estimator's -1 convention consistent with quality checks.
        self.correspondences = self.o3d.core.Tensor(
            np.where(valid, nearest, -1), self.o3d.core.int64, self.device)
        self.pairs = np.column_stack((np.flatnonzero(valid), nearest[valid])).astype(np.int32)
        self.fitness = len(self.pairs) / len(self.reference)
        self.rmse = float(np.sqrt(squared[valid].mean())) if len(self.pairs) else 0.0

    def result(self, failure_reason=''):
        return ICPStep(self.transform.copy(), self.fitness, self.rmse, self.pairs, failure_reason)

    def step(self):
        if len(self.pairs) < 3:
            return self.result('insufficient_correspondences')
        try:
            update = self.estimator.compute_transformation(
                self.source, self.target, self.correspondences).cpu().numpy()
        except RuntimeError as exc:
            # This is a geometric failure, not a CUDA context failure. All other
            # native errors propagate so callers do not reuse a faulted context.
            if 'singular 6x6 linear system' in str(exc).lower():
                return self.result('underconstrained_point_to_plane')
            raise
        self.transform = update @ self.transform
        if not np.isfinite(self.transform).all():
            return self.result('invalid_registration_result')
        self.update_source()
        self.evaluate()  # downloads synchronize GPU work before reporting timings
        return self.result()
