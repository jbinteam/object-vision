"""Persistent GPU point-to-plane ICP; only compact iteration status crosses PCIe.

Triton scans small clouds; Open3D supplies indexed search for larger targets.
CUDA graphs replay fused solve, transform and validation on the default stream.
Float64 and the existing Euler update / convergence thresholds are retained.
"""
import time
import numpy as np


class CudaPlaneEngine:
    def __init__(self, o3d, reference, capacity=131072):
        import torch
        self.torch, self.o3d = torch, o3d
        # Own this allocation in PyTorch: compiled graphs may outlive the caller,
        # so they must not retain memory owned by Open3D's process-exit cacher.
        self.reference = torch.utils.dlpack.from_dlpack(reference.to_dlpack()).clone()
        self.device = self.reference.device
        self.capacity = max(capacity, len(self.reference))
        self.rank_fallbacks = 0
        self.target_count = len(self.reference)
        self.radius = .025
        with torch.cuda.device(self.device), torch.cuda.stream(torch.cuda.default_stream(self.device)):
            self._allocate_and_capture()

    def _allocate_and_capture(self):
        t = self.torch
        options = dict(device=self.device, dtype=t.float64)
        n = len(self.reference)
        self.target = t.zeros((self.capacity, 3), **options)
        self.normals = t.zeros_like(self.target)
        self.target[:n].copy_(self.reference)
        warm_normals = t.sin(self.reference * 100) + 0.3
        self.normals[:n].copy_(warm_normals / t.linalg.vector_norm(warm_normals, dim=1, keepdim=True))
        self.positions = self.reference.clone()
        self.pose = t.eye(4, **options)
        self.update = self.pose.clone()
        self.previous = self.pose.clone()
        self.indices = t.arange(n, device=self.device, dtype=t.int64)
        self.distances = t.zeros(n, **options)
        self.radius_squared = t.full((), self.radius ** 2, **options)
        from .cuda_search import nearest
        nearest(self.positions, self.target, self.indices, self.distances, self.radius_squared, n)
        self.status = t.zeros(9, **options)
        self.eye3, self.eye6 = t.eye(3, **options), t.eye(6, **options)
        self.rotation_tolerance = self.eye3 * 1e-5 + 1e-5
        self.query = self.o3d.core.Tensor.from_dlpack(t.utils.dlpack.to_dlpack(self.positions))
        # Warm cuBLAS/cuSOLVER and allocations outside capture and frame budgets.
        stream = t.cuda.Stream(device=self.device)
        stream.wait_stream(t.cuda.current_stream(self.device))
        with t.cuda.stream(stream):
            for _ in range(3):
                self._solve()
                self._metrics()
        t.cuda.current_stream(self.device).wait_stream(stream)
        self._solve = t.compile(self._solve, fullgraph=True, options={'triton.cudagraphs': False})
        self._metrics = t.compile(self._metrics, fullgraph=True, options={'triton.cudagraphs': False})
        with t.cuda.stream(stream):
            for _ in range(3):
                self._solve()
                self._metrics()
        t.cuda.current_stream(self.device).wait_stream(stream)
        self.solve_graph, self.metrics_graph = t.cuda.CUDAGraph(), t.cuda.CUDAGraph()
        with t.cuda.graph(self.solve_graph):
            self._solve()
        with t.cuda.graph(self.metrics_graph):
            self._metrics()

    def _matches(self):
        valid = self.indices >= 0
        safe_indices = self.indices.clamp_min(0)
        normals = self.normals[safe_indices] * valid[:, None]
        return valid, safe_indices, normals

    def _solve(self):
        t = self.torch
        self.previous.copy_(self.pose)
        valid, indices, normals = self._matches()
        jacobian = t.cat((t.linalg.cross(self.positions, normals), normals), dim=1)
        residual = ((self.positions - self.target[indices]) * normals).sum(dim=1)
        solution, info = t.linalg.solve_ex((jacobian[:, :, None] * jacobian[:, None, :]).sum(dim=0), -(jacobian * residual[:, None]).sum(dim=0), check_errors=False)
        self.solver_info = t.where((info == 0) & ~t.isfinite(solution).all(), -1, info)
        # A singular solve is rejected in status; keep its temporary coordinates
        # finite so nearest-neighbor search is never fed NaNs or infinities.
        solution = t.where((info == 0) & t.isfinite(solution).all(), solution, t.zeros_like(solution))
        sx, sy, sz = t.sin(solution[:3]).unbind()
        cx, cy, cz = t.cos(solution[:3]).unbind()
        rotation = t.stack((cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz,
                            cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-sx*cz,
                            -sy, sx*cy, cx*cy)).reshape(3, 3)
        self.update[:3, :3].copy_(rotation)
        self.update[:3, 3].copy_(solution[3:])
        self.pose.copy_((self.update[:, :, None] * self.previous[None, :, :]).sum(dim=1))
        self.positions.copy_((self.reference[:, None, :] * self.pose[None, :3, :3]).sum(dim=2) + self.pose[:3, 3])

    def _metrics(self):
        t = self.torch
        valid, _, normals = self._matches()
        count = valid.sum().to(t.float64)
        denominator = count.clamp_min(1)
        center = (self.positions * valid[:, None]).sum(dim=0) / denominator
        centered = self.positions - center
        self.centered_jacobian = t.cat((t.linalg.cross(centered, normals), normals), dim=1)
        gram = (self.centered_jacobian[:, :, None] * self.centered_jacobian[:, None, :]).sum(dim=0)
        upper = gram.abs().sum(dim=1).max()  # upper bound on largest eigenvalue
        # Compare sigma_min/sigma_max to 1e-6 using PSD tests. A conservative
        # roundoff band sends ambiguous matrices to the original direct SVD.
        error = (8 * len(self.reference) * t.finfo(t.float64).eps) * upper
        _, pass_info = t.linalg.cholesky_ex(gram - (upper * 1e-12 + error) * self.eye6, check_errors=False)
        rank = t.where(pass_info == 0, 1., 0.)
        rotation = self.pose[:3, :3]
        determinant = t.dot(rotation[0], t.linalg.cross(rotation[1], rotation[2]))
        rigid = (((rotation[:, :, None] * rotation[:, None, :]).sum(dim=0) - self.eye3).abs() <= self.rotation_tolerance).all() & ((determinant - 1).abs() <= 2e-5)
        delta = (self.previous[:3, :3, None] * rotation[:, None, :]).sum(dim=0)
        skew = t.stack((delta[2, 1]-delta[1, 2], delta[0, 2]-delta[2, 0], delta[1, 0]-delta[0, 1]))
        angle = t.atan2(t.linalg.vector_norm(skew) * .5, (delta.diagonal().sum() - 1) * .5).abs()
        translation = t.linalg.vector_norm(self.pose[:3, 3] - self.previous[:3, 3])
        rmse = t.sqrt((self.distances * valid).sum() / denominator)
        finite = t.isfinite(self.pose).all() & t.isfinite(rmse)
        self.status.copy_(t.stack((count, count / len(self.reference), rmse, translation,
                                   angle, rigid.to(t.float64), finite.to(t.float64), rank,
                                   self.solver_info.to(t.float64))))

    def _search(self):
        t = self.torch
        if self.target_count <= 8192:
            from .cuda_search import nearest
            nearest(self.positions, self.target, self.indices, self.distances, self.radius_squared, self.target_count)
            return
        indices, distances, _ = self.search.hybrid_search(self.query, self.radius, 1)
        nearest = t.utils.dlpack.from_dlpack(indices.to_dlpack()).reshape(-1)
        squared = t.utils.dlpack.from_dlpack(distances.to_dlpack()).reshape(-1)
        self.indices.copy_(t.where(squared < self.radius ** 2, nearest, -1))
        self.distances.copy_(squared)

    def register(self, target, initial, config, result, started):
        t = self.torch
        with t.cuda.device(self.device), t.cuda.stream(t.cuda.default_stream(self.device)):
            try:
                return self._register(target, initial, config, result, started)
            finally:
                # Compiled graphs cache this workspace. Do not retain an Open3D
                # allocation until its memory manager has already shut down.
                self.search = None

    def _register(self, target, initial, cfg, result, started):
        t = self.torch
        count = len(target.points)
        if count > self.capacity:
            raise ValueError(f'CUDA target exceeds workspace capacity {self.capacity}')
        icp_start = time.perf_counter()
        if icp_start - started > cfg.icp_time_budget:
            result.reason = 'time_budget_exceeded'
            return result
        self.target[:count].copy_(t.as_tensor(np.asarray(target.points), device=self.device))
        self.normals[:count].copy_(t.as_tensor(np.asarray(target.normals), device=self.device))
        self.pose.copy_(t.as_tensor(initial, device=self.device))
        self.positions.copy_((self.reference[:, None, :] * self.pose[None, :3, :3]).sum(dim=2) + self.pose[:3, 3])
        target_view = self.o3d.core.Tensor.from_dlpack(t.utils.dlpack.to_dlpack(self.target[:count]))
        self.target_count = count
        self.radius = cfg.icp_max_correspondence_distance
        self.radius_squared.fill_(self.radius ** 2)
        if count > 8192:
            self.search = self.o3d.core.nns.NearestNeighborSearch(target_view)
            self.search.hybrid_index(self.radius)
        self._search()
        self.rank_fallbacks = 0
        result.reason = 'iteration_limit'
        for iteration in range(cfg.icp_max_iterations):
            if time.perf_counter() - started > cfg.icp_time_budget:
                result.reason = 'time_budget_exceeded'
                break
            self.solve_graph.replay()
            self._search()
            self.metrics_graph.replay()
            values = self.status.cpu().numpy()  # one 72-byte status download
            matches, fitness, rmse, translation, angle, rigid, finite, rank, solver_info = values
            result.iterations = iteration + 1
            result.correspondences, result.fitness, result.rmse = int(matches), float(fitness), float(rmse)
            if not finite or not rigid:
                result.reason = 'invalid_registration_result'
                break
            if matches < cfg.minimum_correspondences:
                result.reason = 'insufficient_correspondences'
                break
            if solver_info < 0:
                result.reason = 'invalid_registration_result'
                break
            if solver_info:
                result.reason = 'underconstrained_point_to_plane'
                break
            if rank == 0:
                # Rare threshold/degeneracy case: retain the original stable SVD
                # decision instead of relaxing the observability threshold.
                self.rank_fallbacks += 1
                singular = np.linalg.svd(self.centered_jacobian.cpu().numpy(), compute_uv=False)
                rank = 1 if singular[-1] > singular[0] * 1e-6 else -1
            if rank < 0:
                result.reason = 'underconstrained_point_to_plane'
                break
            if translation <= cfg.icp_transformation_epsilon and angle <= cfg.icp_rotation_epsilon:
                result.converged = True
                if fitness < cfg.fitness_threshold:
                    result.reason = 'low_fitness'
                elif rmse > cfg.rmse_threshold:
                    result.reason = 'high_rmse'
                else:
                    result.accepted, result.reason = True, 'accepted'
                break
        result.transform = self.pose.cpu().numpy().copy()  # final pose only
        result.icp_ms = (time.perf_counter() - icp_start) * 1000
        if time.perf_counter() - started > cfg.icp_time_budget:
            result.accepted, result.reason = False, 'time_budget_exceeded'
        return result
