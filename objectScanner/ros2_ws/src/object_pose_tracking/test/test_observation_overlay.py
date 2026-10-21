import numpy as np
import time
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import object_pose_tracking.observation_node as observation_module
from object_pose_tracking.observation_node import draw_detection_box
from object_pose_tracking.ros_support import sensor_qos
from object_pose_tracking.segmentation import Detection


def test_detection_box_draws_on_copy_and_clips_to_image():
    rgb = np.zeros((80, 100, 3), np.uint8)
    detection = Detection(np.ones((80, 100), bool),
                          np.array([-5., 20., 60., 65.]), .93)
    annotated = draw_detection_box(rgb, detection)
    assert annotated.shape == rgb.shape
    assert np.count_nonzero(rgb) == 0
    np.testing.assert_array_equal(annotated[20, 0], [0, 255, 0])
    np.testing.assert_array_equal(annotated[65, 60], [0, 255, 0])


def test_no_detection_keeps_rgb_pixels():
    rgb = np.full((30, 40, 3), [11, 22, 33], np.uint8)
    annotated = draw_detection_box(rgb, None)
    np.testing.assert_array_equal(annotated, rgb)
    assert annotated is not rgb


def test_observation_publishes_annotated_rgb_with_source_header(monkeypatch):
    class FakeSegmenter:
        device = 'cpu'

        def __init__(self, *args):
            pass

        def predict(self, rgb):
            return [Detection(np.ones(rgb.shape[:2], bool), np.array([5., 20., 60., 65.]), .93)]

    class FakePose:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, rgb):
            return []

    monkeypatch.setattr(observation_module, 'YoloSegmenter', FakeSegmenter)
    monkeypatch.setattr(observation_module, 'YoloPoseEstimator', FakePose)
    rclpy.init(args=[])
    node = observation_module.ObservationNode()
    probe = Node('detection_image_probe')
    received = []
    subscription = probe.create_subscription(
        Image, '/object_pose/detection_image', received.append, sensor_qos(2))
    try:
        deadline = time.monotonic() + 3
        while subscription.get_publisher_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=.02)
        assert subscription.get_publisher_count() == 1
        header = Header(stamp=node.get_clock().now().to_msg(),
                        frame_id='camera_color_optical_frame')
        msg = CvBridge().cv2_to_imgmsg(np.zeros((80, 100, 3), np.uint8), encoding='rgb8')
        msg.header = header
        node.image(msg)
        deadline = time.monotonic() + 3
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=.02)
        assert len(received) == 1
        assert received[0].header == header
        assert received[0].encoding == 'rgb8'
        np.testing.assert_array_equal(
            CvBridge().imgmsg_to_cv2(received[0], desired_encoding='rgb8')[20, 5],
            [0, 255, 0])
    finally:
        probe.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
