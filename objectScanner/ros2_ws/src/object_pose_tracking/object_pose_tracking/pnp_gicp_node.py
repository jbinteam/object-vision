"""YOLO keypoints -> robust PnP -> tracked coarse-to-fine GICP -> TF.

Every transform in this node follows:
    P_camera = T_camera_object @ P_object
"""
import time

import cv2
import message_filters
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from rclpy.time import Time

from .gicp import CoarseToFineGICP, GICPConfig, GICPLevel
from .box_overlay import draw_box_wireframe
from .keypoint_messages import read_keypoints
from .pnp import (PnPConfig, RobustPnP, depth_corrected_seed,
                  load_object_keypoints, project_keypoints)
from .pose_filter import PoseEMA, TrackingState
from .pose_outputs import pose_and_transform
from .registration import motion_delta
from .ros_support import MovingAverageMetrics, PipelineNode, run, stamp_ns


def transform_from_tf(message):
    t, q = message.transform.translation, message.transform.rotation
    output = np.eye(4)
    output[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    output[:3, 3] = [t.x, t.y, t.z]
    return output


class PnPGICPNode(PipelineNode):
    def __init__(self):
        super().__init__('pnp_gicp_pose')
        self.camera_frame = self.param('camera_frame', 'camera_color_optical_frame')
        self.object_frame = self.param('object_frame', 'object')
        self.base_frame = self.param('base_frame', 'base_link')
        self.max_age = self.positive('maximum_data_age', .75)
        self.expected_keypoints = int(self.param('expected_keypoints', 8))
        object_points, keypoint_metadata = load_object_keypoints(
            self.param('object_keypoints_path', ''))
        if len(object_points) != self.expected_keypoints:
            raise ValueError('expected_keypoints differs from object_keypoints.yaml')
        if keypoint_metadata.get('object_frame') not in (None, self.object_frame):
            raise ValueError('Configured object_frame differs from object keypoint frame')
        pnp_defaults = vars(PnPConfig())
        self.pnp = RobustPnP(object_points, PnPConfig(**{
            name: self.param(name, default) for name, default in pnp_defaults.items()}))
        levels = tuple(GICPLevel(
            name, self.param(f'gicp_{name}_voxel_size', voxel),
            self.param(f'gicp_{name}_max_correspondence_distance', distance),
            int(self.param(f'gicp_{name}_max_iterations', iterations)),
            self.param(f'gicp_{name}_transformation_epsilon', epsilon),
            self.param(f'gicp_{name}_enabled', True))
            for name, voxel, distance, iterations, epsilon in (
                ('coarse', .010, .035, 25, 1e-5),
                ('medium', .005, .020, 20, 3e-6),
                ('fine', .002, .010, 15, 1e-6)))
        self.gicp = CoarseToFineGICP(
            self.param('reference_cloud_path', ''), GICPConfig(
                levels=levels,
                minimum_points=int(self.param('minimum_points', 30)),
                minimum_correspondences=int(self.param('minimum_correspondences', 30)),
                fitness_threshold=self.param('gicp_fitness_threshold', .20),
                rmse_threshold=self.param('gicp_rmse_threshold', .012),
                overlap_threshold=self.param('gicp_overlap_threshold', .15),
                visibility_filter_enabled=self.param('visibility_filter_enabled', True),
                visibility_normal_facing=self.param('visibility_normal_facing', True),
                tracking_fine_only=self.param('tracking_fine_only', True)))
        self.state = TrackingState(int(self.param('tracking_failures_until_lost', 2)))
        self.filter = PoseEMA(self.param('translation_filter_alpha', .45),
                              self.param('rotation_filter_alpha', .45))
        self.max_translation_jump = self.positive('maximum_translation_jump', .12)
        self.max_rotation_jump = self.positive('maximum_rotation_jump', .8)
        self.max_gicp_reprojection_error = self.positive(
            'max_gicp_keypoint_reprojection_error', 10.)
        self.depth_seed_correction_enabled = self.param(
            'depth_seed_correction_enabled', False)
        self.depth_seed_correction_trigger = self.param(
            'depth_seed_correction_trigger', .02)
        self.depth_seed_correction_maximum = self.param(
            'depth_seed_correction_maximum', .08)
        if not 0 <= self.depth_seed_correction_trigger < self.depth_seed_correction_maximum:
            raise ValueError('Invalid depth seed correction thresholds')
        self.debug_tf = self.param('publish_debug_tf', True)
        self.axis_length = self.positive('axis_length', .06)
        self.bridge = CvBridge()
        self.timings = MovingAverageMetrics()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf = TransformBroadcaster(self)
        qos = self.image_qos(10)
        self.pose_pubs = {name: self.create_publisher(PoseStamped, topic, self.image_qos(3))
                          for name, topic in (
                              ('pnp', '/object_pose/pnp'), ('gicp', '/object_pose/gicp'),
                              ('filtered', '/object_pose/filtered'), ('base', '/object_pose/base'),
                              ('compatibility', '/object/pose'))}
        self.state_pub = self.create_publisher(String, '/object_pose/tracking_state', 10)
        self.overlay_pub = self.create_publisher(Image, '/object_pose/pnp_reprojection', self.image_qos(2))
        self.box_image_pub = self.create_publisher(Image, '/object_pose/box_image', self.image_qos(2))
        self.reference_pnp_pub = self.create_publisher(
            PointCloud2, '/object/reference_cloud_pnp', self.image_qos(2))
        self.reference_gicp_pub = self.create_publisher(
            PointCloud2, '/object/reference_cloud_gicp', self.image_qos(2))
        self.subscribers = [message_filters.Subscriber(self, kind, topic, qos_profile=qos)
                            for kind, topic in (
                                (PointCloud2, '/object/cloud'),
                                (PointCloud2, '/object/keypoints'),
                                (CameraInfo, '/camera/camera_info'),
                                (Image, '/camera/color/image_raw'))]
        self.synchronizer = message_filters.TimeSynchronizer(
            self.subscribers, int(self.param('sync_queue_size', 20)))
        self.synchronizer.registerCallback(self.observe)
        self.last_stamp = -1
        self.status('pnp_gicp_ready', reference_points=len(self.gicp.reference_points),
                    keypoints=len(object_points), transform_convention='P_camera=T_camera_object*P_object')

    def pose_message(self, header, transform, child_frame):
        return pose_and_transform(header, transform, child_frame)

    def publish_pose(self, name, header, transform, child_frame=None, send_tf=False):
        pose, tf = self.pose_message(header, transform, child_frame or self.object_frame)
        self.pose_pubs[name].publish(pose)
        if send_tf:
            self.tf.sendTransform(tf)
        return pose

    def publish_reference(self, publisher, header, transform):
        if not publisher.get_subscription_count():
            return
        points = self.gicp.reference_points @ transform[:3, :3].T + transform[:3, 3]
        publisher.publish(point_cloud2.create_cloud_xyz32(header, points.astype(np.float32)))

    def publish_reprojection(self, image_msg, detected, result):
        if not self.overlay_pub.get_subscription_count():
            return
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8').copy()
        for index, point in enumerate(detected):
            if point[3] and np.isfinite(point[:2]).all():
                cv2.circle(image, tuple(np.rint(point[:2]).astype(int)), 5, (0, 255, 0), 2)
                cv2.putText(image, str(index), tuple(np.rint(point[:2] + [5, -5]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        for point in result.projected_keypoints:
            if np.isfinite(point).all():
                cv2.drawMarker(image, tuple(np.rint(point).astype(int)), (255, 0, 255),
                               cv2.MARKER_CROSS, 9, 2)
        output = self.bridge.cv2_to_imgmsg(image, encoding='rgb8')
        output.header = image_msg.header
        self.overlay_pub.publish(output)

    def publish_box_image(self, image_msg, transform=None, intrinsic=None, distortion=None):
        if not self.box_image_pub.get_subscription_count():
            return
        try:
            rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
            projected = None
            if transform is not None:
                camera_points = self.pnp.object_points @ transform[:3, :3].T + transform[:3, 3]
                if np.all(camera_points[:, 2] > 0):
                    projected = project_keypoints(
                        self.pnp.object_points, transform, intrinsic, distortion)
            output = self.bridge.cv2_to_imgmsg(
                draw_box_wireframe(rgb, projected), encoding='rgb8')
            output.header = image_msg.header
            self.box_image_pub.publish(output)
        except Exception as exc:
            self.get_logger().warning(f'Could not publish 3D box image: {exc}')

    def publish_base_pose(self, header, camera_pose):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, header.frame_id,
                                                 Time.from_msg(header.stamp))
            base_pose = transform_from_tf(tf) @ camera_pose
            base_header = type(header)(stamp=header.stamp, frame_id=self.base_frame)
            self.publish_pose('base', base_header, base_pose)
            return True
        except Exception:
            return False

    def reject(self, reason, image_msg=None, **metrics):
        self.state.reject()
        if self.state.needs_pnp:
            self.filter.reset()
        self.state_pub.publish(String(data=self.state.state))
        if image_msg is not None:
            self.publish_box_image(image_msg)
        self.status(reason, 1, tracking_state=self.state.state, **metrics)

    def observe(self, cloud_msg, keypoint_msg, info, image_msg):
        started = time.perf_counter()
        stamp = stamp_ns(cloud_msg.header.stamp)
        messages = (cloud_msg, keypoint_msg, info, image_msg)
        if stamp <= self.last_stamp or any(not 0 <= self.age(msg.header) <= self.max_age for msg in messages):
            self.status('stale_or_out_of_order_pose_input', 1)
            return
        self.last_stamp, self.last_input = stamp, time.monotonic()
        try:
            if cloud_msg.header.frame_id != self.camera_frame or info.header.frame_id != self.camera_frame:
                raise ValueError('Cloud and CameraInfo must use configured camera_frame')
            points = point_cloud2.read_points_numpy(
                cloud_msg, field_names=['x', 'y', 'z'], skip_nans=True).reshape(-1, 3)
            if len(points) < self.gicp.config.minimum_points:
                self.reject('insufficient_depth_points', image_msg=image_msg, points=len(points))
                return
            keypoints = read_keypoints(keypoint_msg, self.expected_keypoints)
            intrinsic = np.asarray(info.k, dtype=float).reshape(3, 3)
            pnp_result = None
            if self.state.needs_pnp:
                pnp_result = self.pnp.estimate(keypoints, intrinsic, info.d,
                                               observed_centroid=np.median(points, axis=0))
                if any(not 0 <= self.age(message.header) <= self.max_age for message in messages):
                    self.status('pnp_result_expired', 1, pnp_ms=pnp_result.pnp_ms)
                    return
                if not pnp_result.accepted:
                    self.reject('pnp_rejected:' + pnp_result.reason, image_msg=image_msg,
                                valid_keypoints=pnp_result.valid_keypoints,
                                pnp_inliers=pnp_result.inlier_count,
                                reprojection_mean_px=pnp_result.mean_reprojection_error)
                    return
                initial = pnp_result.transform
                self.publish_pose('pnp', cloud_msg.header, initial, 'object_pnp', self.debug_tf)
                self.publish_reference(self.reference_pnp_pub, cloud_msg.header, initial)
                self.publish_reprojection(image_msg, keypoints, pnp_result)
                depth_seed_delta = 0.0
                depth_seed_applied = False
                if self.depth_seed_correction_enabled:
                    initial, depth_seed_delta, depth_seed_applied = depth_corrected_seed(
                        initial, np.median(points, axis=0),
                        self.depth_seed_correction_trigger,
                        self.depth_seed_correction_maximum)
            else:
                initial = self.state.pose.copy()
            result = self.gicp.register(points, initial, intrinsic, (info.height, info.width),
                                        tracking=pnp_result is None)
            metrics = dict(gicp_fitness=result.fitness, gicp_rmse_m=result.rmse,
                           gicp_overlap=result.overlap,
                           gicp_correspondences=result.correspondences,
                           visibility_points=result.visible_reference_points,
                           gicp_total_ms=result.total_ms, **{
                               f'gicp_{name}_ms': value for name, value in result.level_ms.items()})
            if pnp_result is not None:
                metrics.update(depth_seed_delta_m=depth_seed_delta,
                               depth_seed_correction_applied=depth_seed_applied)
            if any(not 0 <= self.age(message.header) <= self.max_age for message in messages):
                self.status('pose_result_expired', 1, **metrics)
                return
            if not result.accepted:
                self.reject('gicp_rejected:' + result.reason, image_msg=image_msg, **metrics)
                return
            if pnp_result is not None:
                gicp_projection = project_keypoints(
                    self.pnp.object_points, result.transform, intrinsic, info.d)
                gicp_reprojection = float(np.mean(np.linalg.norm(
                    gicp_projection[pnp_result.inlier_ids]
                    - keypoints[pnp_result.inlier_ids, :2], axis=1)))
                metrics['gicp_keypoint_reprojection_mean_px'] = gicp_reprojection
                if gicp_reprojection > self.max_gicp_reprojection_error:
                    self.reject('gicp_keypoint_inconsistency', image_msg=image_msg, **metrics)
                    return
            if self.state.pose is not None:
                translation, rotation = motion_delta(self.state.pose, result.transform)
                metrics.update(translation_delta_m=translation, rotation_delta_rad=rotation)
                if translation > self.max_translation_jump or rotation > self.max_rotation_jump:
                    self.reject('temporal_pose_jump', image_msg=image_msg, **metrics)
                    return
            self.state.accept(result.transform)
            filtered = self.filter.update(result.transform)
            self.publish_pose('gicp', cloud_msg.header, result.transform, 'object_gicp', self.debug_tf)
            self.publish_pose('filtered', cloud_msg.header, filtered, 'object_filtered', self.debug_tf)
            self.publish_pose('compatibility', cloud_msg.header, filtered, self.object_frame, True)
            self.publish_reference(self.reference_gicp_pub, cloud_msg.header, result.transform)
            self.publish_box_image(image_msg, filtered, intrinsic, info.d)
            base_available = self.publish_base_pose(cloud_msg.header, filtered)
            self.state_pub.publish(String(data=self.state.state))
            pipeline_ms = (time.perf_counter() - started) * 1000
            timing_values = {key: value for key, value in metrics.items() if key.endswith('_ms')}
            timing_values['pnp_ms'] = 0. if pnp_result is None else pnp_result.pnp_ms
            timing_values['pipeline_total_ms'] = pipeline_ms
            averages = self.timings.update(**timing_values)
            self.status('pose_accepted', tracking_state=self.state.state,
                        initializer='pnp' if pnp_result is not None else 'previous_pose',
                        pnp_inliers=0 if pnp_result is None else pnp_result.inlier_count,
                        reprojection_mean_px=(float('nan') if pnp_result is None
                                              else pnp_result.mean_reprojection_error),
                        base_pose_available=base_available, **metrics,
                        pnp_ms=timing_values['pnp_ms'], pipeline_total_ms=pipeline_ms,
                        **averages)
        except Exception as exc:
            self.reject(f'pnp_gicp_error:{exc}', image_msg=image_msg)


def main(args=None):
    run(PnPGICPNode, args)
