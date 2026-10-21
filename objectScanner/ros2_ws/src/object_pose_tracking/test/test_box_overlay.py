import numpy as np
from types import SimpleNamespace
from cv_bridge import CvBridge
from std_msgs.msg import Header

from object_pose_tracking.box_overlay import CUBOID_EDGES, draw_box_wireframe
from object_pose_tracking.pnp import project_keypoints
from object_pose_tracking.pnp_gicp_node import PnPGICPNode


def test_wireframe_connects_all_eight_ordered_corners_without_changing_input():
    assert len(CUBOID_EDGES) == 12
    assert all(bin(a ^ b).count('1') == 1 for a, b in CUBOID_EDGES)
    rgb = np.zeros((160, 160, 3), np.uint8)
    corners = np.array([[20, 20], [80, 20], [20, 80], [80, 80],
                        [40, 40], [100, 40], [40, 100], [100, 100]], float)
    image = draw_box_wireframe(rgb, corners)
    assert not np.any(rgb)
    assert image[20, 50, 1] > 0  # first horizontal edge
    assert image[50, 20, 1] > 0  # first vertical edge
    assert image[30, 30, 1] > 0  # corresponding corners across depth


def test_rejected_pose_has_no_wireframe():
    rgb = np.zeros((160, 160, 3), np.uint8)
    image = draw_box_wireframe(rgb)
    assert image[80, 80].sum() == 0
    assert np.any(image[:35, :250])  # status text appears at the top


def test_published_image_projects_calibrated_corners_with_source_header():
    corners = np.array([[x, y, z] for z in (-.05, .05)
                        for y in (-.05, .05) for x in (-.05, .05)])
    transform = np.eye(4)
    transform[2, 3] = .7
    intrinsic = np.array([[300., 0., 80.], [0., 300., 80.], [0., 0., 1.]])
    bridge = CvBridge()
    source = bridge.cv2_to_imgmsg(np.zeros((160, 160, 3), np.uint8), encoding='rgb8')
    source.header = Header(frame_id='camera_color_optical_frame')
    published = []
    fake = SimpleNamespace(bridge=bridge, pnp=SimpleNamespace(object_points=corners),
                           box_image_pub=SimpleNamespace(publish=published.append, get_subscription_count=lambda: 1),
                           get_logger=lambda: SimpleNamespace(warning=lambda message: None))
    PnPGICPNode.publish_box_image(fake, source, transform, intrinsic, np.zeros(5))
    assert len(published) == 1
    assert published[0].header == source.header
    assert published[0].encoding == 'rgb8'
    expected = draw_box_wireframe(np.zeros((160, 160, 3), np.uint8),
                                  project_keypoints(corners, transform, intrinsic, np.zeros(5)))
    np.testing.assert_array_equal(bridge.imgmsg_to_cv2(published[0], desired_encoding='rgb8'),
                                  expected)
