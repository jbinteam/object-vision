"""Pose, TF, and expiring RViz axes from one accepted absolute transform."""
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.duration import Duration
from .registration import is_rigid


def pose_and_transform(header, transform, object_frame):
    """Create required pose/TF messages without constructing optional RViz markers."""
    if not is_rigid(transform):
        raise ValueError('Cannot publish a nonrigid pose')
    position, quaternion = transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()
    pose = PoseStamped(header=header)
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, position)
    pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = map(float, quaternion)
    tf = TransformStamped(header=header, child_frame_id=object_frame)
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, position)
    tf.transform.rotation = pose.pose.orientation
    return pose, tf


def pose_and_axes(header, transform, object_frame, axis_length, lifetime):
    pose, tf = pose_and_transform(header, transform, object_frame)
    position = transform[:3, 3]
    markers = []
    for index, color in enumerate([(1., 0., 0.), (0., 1., 0.), (0., .4, 1.)]):
        marker = Marker(header=header, ns='icp_object_frame', id=index,
                        type=Marker.ARROW, action=Marker.ADD)
        marker.pose = pose.pose
        endpoint = np.eye(3)[index] * axis_length
        marker.points = [Point(), Point(x=float(endpoint[0]), y=float(endpoint[1]), z=float(endpoint[2]))]
        marker.scale.x = axis_length * .045
        marker.scale.y = axis_length * .10
        marker.scale.z = axis_length * .18
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.)
        marker.lifetime = Duration(seconds=lifetime).to_msg()
        markers.append(marker)
    label = Marker(header=header, ns='icp_object_frame', id=3,
                   type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
    label.text = object_frame
    label_point = position + transform[:3, :3] @ np.array([0., 0., axis_length * 1.15])
    label.pose.position.x, label.pose.position.y, label.pose.position.z = map(float, label_point)
    label.pose.orientation.w = 1.
    label.scale.z = axis_length * .22
    label.color = ColorRGBA(r=1., g=1., b=1., a=1.)
    label.lifetime = Duration(seconds=lifetime).to_msg()
    markers.append(label)
    return pose, tf, MarkerArray(markers=markers)


def clear_axes(header):
    return MarkerArray(markers=[Marker(header=header, ns='icp_object_frame', id=i,
                                      action=Marker.DELETE) for i in range(4)])
