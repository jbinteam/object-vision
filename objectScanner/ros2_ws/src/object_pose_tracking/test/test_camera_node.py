"""Camera bridge selection and real ROS messages, without starting a device."""
from pathlib import Path
import time
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy

from object_pose_tracking import camera_node


@pytest.fixture
def camera_bridge(monkeypatch):
    project = Path(__file__).resolve().parents[4]
    rclpy.init(args=["--ros-args", "-p", f"project_root:={project}",
                    "-p", "camera_backend:=orbbec", "-p", "serial:=gemini-serial"])
    worker = Mock()
    worker.poll.return_value = 0
    popen = Mock(return_value=worker)
    monkeypatch.setattr(camera_node.subprocess, "Popen", popen)
    monkeypatch.setattr(camera_node.threading, "Thread", Mock())
    node = camera_node.CameraNode()
    node.color_pub = Mock()
    node.depth_pub = Mock()
    node.info_pub = Mock()
    try:
        yield node, popen
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_camera_backend_is_forwarded_and_ros_environment_is_isolated(camera_bridge):
    _, popen = camera_bridge
    command = popen.call_args.args[0]
    assert command[command.index("--camera-backend") + 1] == "orbbec"
    assert command[command.index("--serial") + 1] == "gemini-serial"
    environment = popen.call_args.kwargs["env"]
    assert "PYTHONPATH" not in environment and "PYTHONHOME" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


def test_camera_bridge_preserves_rgb_mm_and_rectified_intrinsics(camera_bridge):
    node, _ = camera_bridge
    rgb = np.full((2, 3, 3), [255, 64, 0], np.uint8)
    depth = np.full((2, 3), 725, np.uint16)
    intrinsic = np.array([[100., 0., 1.], [0., 120., .5], [0., 0., 1.]])
    timestamp = node.get_clock().now().nanoseconds
    node.latest = rgb, depth, intrinsic, timestamp
    node.last_received = time.monotonic()
    node.received = True
    node.publish_latest()
    color = node.color_pub.publish.call_args.args[0]
    depth_message = node.depth_pub.publish.call_args.args[0]
    info = node.info_pub.publish.call_args.args[0]
    assert color.encoding == "rgb8" and depth_message.encoding == "16UC1"
    np.testing.assert_array_equal(node.bridge.imgmsg_to_cv2(color), rgb)
    np.testing.assert_array_equal(node.bridge.imgmsg_to_cv2(depth_message), depth)
    assert color.header == depth_message.header == info.header
    assert color.header.frame_id == "camera_color_optical_frame"
    assert color.header.stamp.sec * 10**9 + color.header.stamp.nanosec == timestamp
    np.testing.assert_array_equal(info.k, intrinsic.ravel())
    np.testing.assert_array_equal(info.d, np.zeros(5))
    np.testing.assert_array_equal(info.p, [100., 0., 1., 0., 0., 120., .5, 0., 0., 0., 1., 0.])
    assert (info.width, info.height) == (3, 2)
