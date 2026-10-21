"""Real ROS messages/DDS + exact sync, without camera or inference dependencies."""
import time
import numpy as np
import pytest
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from object_pose_tracking.cloud_node import CloudNode
from object_pose_tracking.ros_support import sensor_qos


@pytest.fixture
def pipeline():
    rclpy.init(args=[])
    cloud = CloudNode()
    probe = Node('phase1_test_probe')
    executor = SingleThreadedExecutor()
    executor.add_node(cloud)
    executor.add_node(probe)
    outputs = []
    sub = probe.create_subscription(PointCloud2, '/object/cloud', outputs.append, sensor_qos(10))
    pubs = [probe.create_publisher(kind, topic, sensor_qos(10)) for kind, topic in
            [(Image, '/object/mask'), (Image, '/camera/depth/image_raw'), (CameraInfo, '/camera/camera_info'),
             (Image, '/camera/color/image_raw')]]
    def spin(duration=.1):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.01)
    deadline = time.monotonic() + 3
    while (any(p.get_subscription_count() == 0 for p in pubs) or sub.get_publisher_count() == 0) and time.monotonic() < deadline:
        spin(.05)
    assert all(p.get_subscription_count() > 0 for p in pubs)
    yield cloud, probe, pubs, outputs, spin
    executor.shutdown()
    cloud.destroy_node()
    probe.destroy_node()
    rclpy.shutdown()


def messages(probe):
    bridge = CvBridge()
    header = Header(stamp=probe.get_clock().now().to_msg(), frame_id='camera_color_optical_frame')
    mask = bridge.cv2_to_imgmsg(np.ones((8,8),np.uint8)*255, encoding='mono8')
    depth = bridge.cv2_to_imgmsg(np.ones((8,8),np.uint16)*1000, encoding='16UC1')
    mask.header = depth.header = header
    info = CameraInfo(header=header, width=8, height=8)
    info.k = [100.,0.,3.5,0.,100.,3.5,0.,0.,1.]
    info.d = [0.] * 5
    rgb = np.zeros((8,8,3), np.uint8)
    rgb[:, :4] = [255, 0, 0]
    rgb[:, 4:] = [0, 128, 255]
    color = bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
    color.header = header
    return mask, depth, info, color


def test_exact_sync_preserves_measurement_and_frame(pipeline):
    cloud, probe, pubs, outputs, spin = pipeline
    msgs = messages(probe)
    pubs[0].publish(msgs[0]); pubs[1].publish(msgs[1])
    spin(.1)
    assert not outputs  # no calibration for this timestamp yet
    pubs[2].publish(msgs[2])
    spin(.05)
    assert not outputs  # RGB from this acquisition is also required
    pubs[3].publish(msgs[3])
    spin(.15)
    assert len(outputs) == 1
    output = outputs[0]
    assert output.header == msgs[1].header
    points = point_cloud2.read_points_numpy(output, field_names=['x', 'y', 'z'])
    assert len(points) >= 30
    np.testing.assert_allclose(points[:,2], 1.)
    assert points[:,0].min() < 0 < points[:,0].max()
    packed = point_cloud2.read_points(output)['rgb']
    np.testing.assert_array_equal(packed[points[:,0] < 0], 0xFF0000)
    np.testing.assert_array_equal(packed[points[:,0] > 0], 0x0080FF)
    # Exact synchronization does not pair a different mask acquisition time.
    msgs = messages(probe)
    msgs[0].header = Header(stamp=probe.get_clock().now().to_msg(), frame_id=msgs[1].header.frame_id)
    for pub, msg in zip(pubs,msgs): pub.publish(msg)
    spin(.15)
    assert len(outputs) == 1
    # A color frame from another acquisition cannot color the current cloud.
    msgs = messages(probe)
    msgs[3].header = Header(stamp=probe.get_clock().now().to_msg(), frame_id=msgs[1].header.frame_id)
    for pub, msg in zip(pubs,msgs): pub.publish(msg)
    spin(.1)
    assert len(outputs) == 1


def test_empty_mask_clears_cloud(pipeline):
    cloud, probe, pubs, outputs, spin = pipeline
    msgs = messages(probe)
    msgs[0].data = bytes(64)
    cloud.extract(*msgs)
    spin()
    assert len(outputs) == 1 and outputs[0].width == 0


@pytest.mark.parametrize('failure', ['frame', 'distortion', 'dimensions', 'missing_tf', 'color_dimensions', 'color_frame'])
def test_bad_calibration_or_missing_tf_suppresses_points(pipeline, failure):
    cloud, probe, pubs, outputs, spin = pipeline
    msgs = messages(probe)
    if failure == 'frame':
        msgs[2].header = Header(stamp=msgs[1].header.stamp, frame_id='wrong_optical_frame')
    elif failure == 'distortion':
        msgs[2].d = [0.1,0.,0.,0.,0.]
    elif failure == 'dimensions':
        msgs[2].width = 100
    elif failure == 'color_dimensions':
        msgs[3].width = 7
    elif failure == 'color_frame':
        msgs[3].header = Header(stamp=msgs[1].header.stamp, frame_id='wrong_rgb_frame')
    else:
        cloud.frame = 'robot_base'
    cloud.extract(*msgs)
    spin()
    assert len(outputs) == 1 and outputs[0].width == 0


def test_stale_input_no_fresh_publication(pipeline):
    cloud, probe, pubs, outputs, spin = pipeline
    msgs = messages(probe)
    msgs[1].header.stamp.sec -= 5
    cloud.extract(*msgs)
    spin()
    assert not outputs
