"""Avoid optional work while preserving measurements and accepted tracking state."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from object_pose_tracking.gicp import GICPResult
from object_pose_tracking.keypoint_messages import create_keypoints, read_keypoints
from object_pose_tracking.observation_node import ObservationNode
from object_pose_tracking.pnp_gicp_node import PnPGICPNode
from object_pose_tracking.pose_filter import PoseEMA, TrackingState
from object_pose_tracking.pose_outputs import pose_and_transform
from object_pose_tracking.ros_support import MovingAverageMetrics
from object_pose_tracking.segmentation import Detection, InstanceSelector, PoseDetection


def test_initializer_skips_unusable_pose_inference_and_recovers_next_detection():
    bridge = CvBridge()
    rgb = np.zeros((40, 50, 3), np.uint8)
    detection = Detection(np.ones(rgb.shape[:2], bool), np.array([0., 0., 49., 39.]), .9)
    keypoints = np.column_stack((np.arange(16).reshape(8, 2) + 1., np.ones((8, 2))))
    segmenter = Mock()
    segmenter.predict.side_effect = [[], [detection]]
    pose = Mock()
    pose.predict.return_value = [PoseDetection(detection.box, .9, keypoints)]
    fake = SimpleNamespace(
        bridge=bridge, segmenter=segmenter, pose=pose, selector=InstanceSelector(),
        pose_only_for_reinitialization=True, tracking_state='INITIALIZATION',
        minimum_pair_iou=.1, expected_keypoints=8, max_age=.75, last_stamp=-1,
        age=lambda header: .01, mask_pub=Mock(), keypoint_pub=Mock(),
        detection_image_pub=Mock(), status=Mock(), timings=MovingAverageMetrics())
    fake.detection_image_pub.get_subscription_count.return_value = 0
    message = bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
    message.header = Header(frame_id='camera_color_optical_frame')
    message.header.stamp.sec = 1
    ObservationNode.image(fake, message)
    pose.predict.assert_not_called()
    first_mask = fake.mask_pub.publish.call_args.args[0]
    assert not np.any(bridge.imgmsg_to_cv2(first_mask))
    first_keypoints = read_keypoints(fake.keypoint_pub.publish.call_args.args[0], 8)
    assert np.isnan(first_keypoints[:, :2]).all() and not np.any(first_keypoints[:, 3])
    assert not fake.status.call_args.kwargs['pose_inference_ran']
    message.header.stamp.sec = 2
    ObservationNode.image(fake, message)
    pose.predict.assert_called_once()
    np.testing.assert_array_equal(
        read_keypoints(fake.keypoint_pub.publish.call_args.args[0], 8), keypoints)
    assert fake.keypoint_pub.publish.call_args.args[0].header == message.header
    assert fake.status.call_args.kwargs['pose_inference_ran']


def test_unsubscribed_debug_outputs_do_not_access_image_or_reference():
    reference, overlay, box = Mock(), Mock(), Mock()
    for publisher in (reference, overlay, box):
        publisher.get_subscription_count.return_value = 0
    # Deliberately omit gicp/bridge/pnp: these must not be accessed without readers.
    fake = SimpleNamespace(overlay_pub=overlay, box_image_pub=box)
    PnPGICPNode.publish_reference(fake, reference, Header(), np.eye(4))
    PnPGICPNode.publish_reprojection(fake, None, None, None)
    PnPGICPNode.publish_box_image(fake, None)
    for publisher in (reference, overlay, box):
        publisher.publish.assert_not_called()


def test_required_pose_transform_never_constructs_rviz_markers(monkeypatch):
    import object_pose_tracking.pose_outputs as outputs
    marker = Mock(side_effect=AssertionError('Optional RViz markers were constructed'))
    monkeypatch.setattr(outputs, 'Marker', marker)
    transform = np.eye(4)
    transform[:3, 3] = [.01, -.02, .7]
    header = Header(frame_id='camera_color_optical_frame')
    pose, tf = pose_and_transform(header, transform, 'box_center')
    assert pose.header == tf.header == header and tf.child_frame_id == 'box_center'
    assert pose.pose.position.z == tf.transform.translation.z == .7
    assert pose.pose.orientation == tf.transform.rotation
    marker.assert_not_called()


@pytest.mark.parametrize("initialization", [False, True])
def test_slow_pose_compute_does_not_accept_expired_pose_or_update_filter(initialization):
    header = Header(frame_id='camera_color_optical_frame')
    header.stamp.sec = 1
    transform = np.eye(4)
    transform[2, 3] = .7
    state = TrackingState()
    if not initialization:
        state.accept(transform)
    filtering = PoseEMA()
    filtering.update(transform)
    before_state = None if initialization else state.pose.copy()
    before_filter = filtering.value.copy()
    result = GICPResult(transform=transform.copy(), accepted=True, converged=True,
                        reason='accepted', fitness=1., rmse=.001, overlap=1.,
                        correspondences=40, total_ms=800., level_ms={'fine': 800.})
    fake = SimpleNamespace(
        last_stamp=-1, max_age=.75, camera_frame=header.frame_id, expected_keypoints=8,
        age=Mock(side_effect=[.01, .01, .01, .01, .81]), state=state, filter=filtering,
        pnp=SimpleNamespace(estimate=Mock(return_value=SimpleNamespace(accepted=True, pnp_ms=800.))),
        gicp=SimpleNamespace(config=SimpleNamespace(minimum_points=30),
                            register=Mock(return_value=result)),
        status=Mock(), reject=Mock(), publish_pose=Mock(), publish_reference=Mock(),
        publish_box_image=Mock(), publish_base_pose=Mock(), state_pub=Mock())
    rng = np.random.default_rng(45)
    points = rng.normal(size=(40, 3)) * .01 + [0., 0., .7]
    cloud = point_cloud2.create_cloud_xyz32(header, points.astype(np.float32))
    keypoints = create_keypoints(header, np.ones((8, 4), np.float32))
    info = CameraInfo(header=header, width=50, height=40)
    info.k = [100., 0., 25., 0., 100., 20., 0., 0., 1.]
    image = CvBridge().cv2_to_imgmsg(np.zeros((40, 50, 3), np.uint8), encoding='rgb8')
    image.header = header
    PnPGICPNode.observe(fake, cloud, keypoints, info, image)
    if initialization:
        fake.pnp.estimate.assert_called_once()
        fake.gicp.register.assert_not_called()
        assert fake.status.call_args.args[0] == 'pnp_result_expired'
        assert state.pose is None
    else:
        fake.gicp.register.assert_called_once()
        assert fake.status.call_args.args[0] == 'pose_result_expired'
        np.testing.assert_array_equal(state.pose, before_state)
    np.testing.assert_array_equal(filtering.value, before_filter)
    assert state.state == ('INITIALIZATION' if initialization else 'TRACKING')
    assert state.failures == 0
    fake.reject.assert_not_called()
    fake.publish_pose.assert_not_called()
    fake.publish_reference.assert_not_called()
    fake.publish_box_image.assert_not_called()
    fake.state_pub.publish.assert_not_called()
