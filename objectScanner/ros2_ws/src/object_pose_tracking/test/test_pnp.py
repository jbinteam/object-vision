import numpy as np
import pytest
import cv2

from object_pose_tracking.pnp import (PnPConfig, RobustPnP, depth_corrected_seed,
                                      load_object_keypoints, project_keypoints)
from object_pose_tracking.registration import motion_delta


POINTS = np.array([[x, y, z] for z in (-.04, .04) for y in (-.03, .03)
                   for x in (-.05, .05)], dtype=float)
K = np.array([[615., 0, 320], [0, 614., 240], [0, 0, 1.]])


def pose():
    transform = np.eye(4)
    transform[:3, :3], _ = cv2.Rodrigues(np.array([.18, -.25, .12]))
    transform[:3, 3] = [.04, -.02, .65]
    return transform


def detections(transform=None):
    pixels = project_keypoints(POINTS, transform or pose(), K)
    return np.column_stack((pixels, np.full(8, .9), np.ones(8)))


def test_pnp_recovers_object_to_camera_and_rejects_one_outlier():
    detected = detections()
    detected[2, :2] += [90, -70]
    result = RobustPnP(POINTS).estimate(detected, K, observed_centroid=pose()[:3, 3])
    assert result.accepted, result
    distance, angle = motion_delta(pose(), result.transform)
    assert distance < 1e-5 and angle < 1e-4
    assert result.inlier_count == 7 and 2 not in result.inlier_ids
    assert result.mean_reprojection_error < 1e-3


def test_pnp_filters_confidence_and_applies_depth_gate():
    detected = detections()
    detected[:5, 2] = .1
    assert RobustPnP(POINTS).estimate(detected, K).reason == 'insufficient_visible_keypoints'
    result = RobustPnP(POINTS, PnPConfig(max_depth_translation_error=.02)).estimate(
        detections(), K, observed_centroid=[.3, .1, 1.0])
    assert not result.accepted and result.reason == 'pnp_depth_inconsistency'


def test_keypoint_yaml_requires_deterministic_ids(tmp_path):
    good = tmp_path / 'good.yaml'
    good.write_text('units: m\nobject_keypoints:\n' + ''.join(
        f'  - {{id: {i}, xyz: [{x}, {y}, {z}]}}\n' for i, (x, y, z) in enumerate(POINTS)))
    loaded, metadata = load_object_keypoints(good)
    np.testing.assert_allclose(loaded, POINTS)
    bad = tmp_path / 'bad.yaml'
    bad.write_text(good.read_text().replace('id: 3', 'id: 9'))
    with pytest.raises(ValueError, match='contiguous'):
        load_object_keypoints(bad)


def test_depth_corrected_seed_only_changes_material_bounded_z_error():
    original = pose()
    unchanged, delta, applied = depth_corrected_seed(
        original, [0., 0., original[2, 3] + .005])
    np.testing.assert_array_equal(unchanged, original)
    assert not applied and np.isclose(delta, .005)

    corrected, delta, applied = depth_corrected_seed(
        original, [0., 0., original[2, 3] - .035])
    assert applied and np.isclose(delta, -.035)
    np.testing.assert_allclose(corrected[:2, 3], original[:2, 3])
    np.testing.assert_allclose(corrected[:3, :3], original[:3, :3])
    assert np.isclose(corrected[2, 3], original[2, 3] - .035)

    rejected, _, applied = depth_corrected_seed(
        original, [0., 0., original[2, 3] + .2])
    np.testing.assert_array_equal(rejected, original)
    assert not applied
