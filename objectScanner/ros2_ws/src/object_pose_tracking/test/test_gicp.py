import numpy as np
import open3d as o3d

from object_pose_tracking.gicp import (CoarseToFineGICP, GICPConfig, GICPLevel,
                                       ReferenceVisibilityFilter)
from object_pose_tracking.registration import motion_delta


def save_cloud(path, points):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    assert o3d.io.write_point_cloud(str(path), cloud)


def test_visibility_filter_fov_and_facing():
    points = np.array([[0, 0, 1], [2, 0, 1], [0, 0, -1.]])
    normals = np.array([[0, 0, -1.], [0, 0, -1.], [0, 0, 1.]])
    ids = ReferenceVisibilityFilter().filter(
        points, np.eye(4), [[100, 0, 50], [0, 100, 50], [0, 0, 1]], (100, 100), normals)
    np.testing.assert_array_equal(ids, [0])


def test_coarse_to_fine_gicp_refines_absolute_transform(tmp_path):
    rng = np.random.default_rng(8)
    # Three nonparallel faces avoid the planar ambiguity of a single view.
    uv = rng.uniform(-.05, .05, (500, 2))
    points = np.vstack((np.column_stack((np.full(500, -.04), uv)),
                        np.column_stack((uv[:, 0], np.full(500, .035), uv[:, 1])),
                        np.column_stack((uv, np.full(500, .045)))))
    path = tmp_path / 'reference.ply'
    save_cloud(path, points)
    expected = np.eye(4)
    expected[:3, 3] = [.03, -.015, .65]
    observed = points @ expected[:3, :3].T + expected[:3, 3]
    seed = expected.copy(); seed[:3, 3] += [.006, -.004, .005]
    config = GICPConfig(levels=(GICPLevel('test', .003, .025, 50),),
                        visibility_filter_enabled=False, fitness_threshold=.8,
                        overlap_threshold=.8, rmse_threshold=.004)
    result = CoarseToFineGICP(path, config).register(observed, seed)
    assert result.accepted, result
    distance, angle = motion_delta(expected, result.transform)
    assert distance < 1e-3 and angle < 1e-2


def test_tracking_can_select_fine_level_only(tmp_path):
    rng = np.random.default_rng(9)
    points = rng.normal(size=(500, 3)) * [.03, .02, .01]
    path = tmp_path / 'reference.ply'
    save_cloud(path, points)
    levels = (GICPLevel('coarse', .01, .03, 5), GICPLevel('fine', .003, .01, 5))
    registration = CoarseToFineGICP(
        path, GICPConfig(levels=levels, visibility_filter_enabled=False,
                         fitness_threshold=0, overlap_threshold=0, rmse_threshold=1))
    result = registration.register(points, np.eye(4), tracking=True)
    assert result.accepted and set(result.level_ms) == {'fine'}
