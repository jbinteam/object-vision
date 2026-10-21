import numpy as np
import open3d as o3d
import pytest

pytestmark = pytest.mark.skipif(not o3d.core.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('count', [33, 1031, 5000])
def test_tiled_search_matches_exact_float64_distances(count):
    import torch
    from object_pose_tracking.cuda_search import nearest
    rng = np.random.default_rng(15)
    points = rng.normal(0, .03, (count, 3))
    queries = rng.normal(0, .03, (21, 3))
    queries[-1] = [10, 10, 10]  # no correspondence
    target = torch.zeros((count + 123, 3), device='cuda', dtype=torch.float64)
    target[:count].copy_(torch.as_tensor(points, device='cuda'))
    query = torch.as_tensor(queries, device='cuda')
    indices = torch.empty(21, device='cuda', dtype=torch.int64)
    distances = torch.empty(21, device='cuda', dtype=torch.float64)
    radius_squared = torch.tensor(.025 ** 2, device='cuda', dtype=torch.float64)
    nearest(query, target, indices, distances, radius_squared, count)
    squared = ((queries[:, None] - points) ** 2).sum(axis=2)
    expected = squared.argmin(axis=1)
    best = squared[np.arange(21), expected]
    valid = best < .025 ** 2
    np.testing.assert_array_equal(indices.cpu().numpy(), np.where(valid, expected, -1))
    np.testing.assert_allclose(distances.cpu().numpy(), np.where(valid, best, 0), rtol=1e-12, atol=1e-16)


def test_search_ties_and_strict_radius_boundary():
    import torch
    from object_pose_tracking.cuda_search import nearest
    points = torch.tensor([[.01, 0, 0], [-.01, 0, 0], [1.025, 0, 0]], device='cuda', dtype=torch.float64)
    queries = torch.tensor([[0., 0, 0], [1., 0, 0]], device='cuda', dtype=torch.float64)
    # Use the represented target/query difference so the boundary is exact.
    radius = (points[2, 0] - queries[1, 0]).square()
    indices = torch.empty(2, device='cuda', dtype=torch.int64)
    distances = torch.empty(2, device='cuda', dtype=torch.float64)
    nearest(queries, points, indices, distances, radius, 3)
    np.testing.assert_array_equal(indices.cpu().numpy(), [0, -1])
