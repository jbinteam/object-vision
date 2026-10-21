"""Fixed-reference ICP with independent (Phase 2) or previous-pose (Phase 3) seeds."""
import time
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray
from .registration import ICPConfig, FixedReferenceICP, initial_rotation, motion_delta
from .tracking import AcceptedPoseSeed
from .pose_outputs import pose_and_axes, clear_axes
from .ros_support import PipelineNode, run, stamp_ns


class PoseNode(PipelineNode):
    def __init__(self):
        super().__init__('icp_pose')
        self.config = ICPConfig(**{name: self.param(name, default) for name, default in vars(ICPConfig()).items()})
        self.camera_frame = self.param('camera_frame', 'camera_color_optical_frame')
        self.object_frame = self.param('object_frame', 'object')
        if (not self.camera_frame or not self.object_frame or self.camera_frame == self.object_frame
                or self.camera_frame.startswith('/') or self.object_frame.startswith('/')):
            raise ValueError('camera_frame and object_frame must be distinct, nonempty relative frame names')
        self.max_age = self.positive('maximum_data_age', 0.75)
        self.tracking_enabled = self.param('tracking_enabled', False)
        self.tracking_seed = AcceptedPoseSeed(self.positive('tracking_seed_timeout', 0.75))
        self.axis_length = self.positive('axis_length', 0.06)
        self.visual_lifetime = self.positive('visualization_lifetime', 0.5)
        self.debug_period = 1. / self.positive('reference_publish_rate', 10.)
        self.maximum_points = int(self.param('maximum_observed_points', 120000))
        if self.maximum_points < self.config.minimum_points:
            raise ValueError('maximum_observed_points is smaller than minimum_points')
        reference_path = self.param('reference_cloud_path', '')
        self.seed_rotation = initial_rotation(reference_path, self.param('initial_rotation_mode', 'identity'),
                                               self.param('initial_rotation_deg', [0., 0., 0.]))
        self.registration = FixedReferenceICP(reference_path, self.config)
        self.pose_pub = self.create_publisher(PoseStamped, '/object/pose', self.image_qos(1))
        self.overlay_pub = self.create_publisher(PointCloud2, '/object/reference_cloud_transformed', self.image_qos(1))
        self.axes_pub = self.create_publisher(MarkerArray, '/object/frame_axes', self.image_qos(1))
        self.tf = TransformBroadcaster(self)
        self.subscription = self.create_subscription(PointCloud2, '/object/cloud', self.observe, self.image_qos(1))
        self.last_stamp = -1
        self.accepted_stamp = None
        self.visible = False
        self.last_overlay = -float('inf')
        self.create_timer(.1, self.check_timeout)
        self.status('icp_ready', reference_points=len(self.registration.reference_points),
                    icp_backend=self.config.icp_backend, icp_cuda_device=self.config.icp_cuda_device,
                    initialization='previous_accepted_pose' if self.tracking_enabled else 'independent_centroid_each_observation')

    def clear_visuals(self, header=None):
        if self.visible:
            header = header or Header(stamp=self.get_clock().now().to_msg(), frame_id=self.camera_frame)
            self.axes_pub.publish(clear_axes(header))
            self.overlay_pub.publish(point_cloud2.create_cloud_xyz32(header, np.empty((0, 3), np.float32)))
            self.visible = False
        self.accepted_stamp = None

    def check_timeout(self):
        now = self.get_clock().now().nanoseconds
        if self.tracking_seed.expire(now):
            self.status('tracking_seed_expired', 1)
        if self.accepted_stamp is not None and not 0 <= (now - self.accepted_stamp) / 1e9 <= self.max_age:
            self.clear_visuals()
            self.status('pose_expired', 1)
        elif time.monotonic() - self.last_input > self.max_age:
            self.status('no_recent_object_cloud', 1)

    def observe(self, msg):
        start = time.perf_counter()
        stamp = stamp_ns(msg.header.stamp)
        if stamp <= self.last_stamp:
            return
        self.last_stamp, self.last_input = stamp, time.monotonic()
        if not 0 <= self.age(msg.header) <= self.max_age:
            self.clear_visuals()
            self.status('stale_object_cloud', 1)
            return
        try:
            if msg.header.frame_id != self.camera_frame:
                raise ValueError('Cloud frame differs from camera_frame; transform it in the cloud node first')
            if not self.config.minimum_points <= msg.width * msg.height <= self.maximum_points:
                self.clear_visuals()
                self.status('invalid_cloud_point_count', 1, points=msg.width * msg.height)
                return
            points = point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=False).reshape(-1, 3)
            now = self.get_clock().now().nanoseconds
            seed = self.tracking_seed.initial_guess(stamp, now) if self.tracking_enabled else None
            seed_age_ms = (now - self.tracking_seed.stamp_ns) / 1e6 if seed is not None else 0.0
            result = self.registration.register(points, rotation=self.seed_rotation, initial_guess=seed)
            metrics = dict(icp_backend=self.config.icp_backend, fitness=result.fitness, rmse_m=result.rmse,
                           icp_cuda_device=self.config.icp_cuda_device,
                           correspondence_ratio=result.fitness, correspondences=result.correspondences,
                           converged=result.converged, iterations=result.iterations,
                           preparation_ms=result.preparation_ms, icp_ms=result.icp_ms,
                           initial_guess='previous_pose' if seed is not None else 'centroid',
                           tracking_seed_age_ms=seed_age_ms)
            if not result.accepted:
                self.clear_visuals()
                self.status('icp_rejected:' + result.reason, 1, **metrics)
                return
            age = self.age(msg.header)
            if not 0 <= age < self.max_age:
                self.clear_visuals()
                self.status('icp_result_expired', 1, **metrics)
                return
            pose, tf, axes = pose_and_axes(msg.header, result.transform, self.object_frame,
                                           self.axis_length, min(self.visual_lifetime, self.max_age - age))
            if seed is not None:
                delta_translation, delta_rotation = motion_delta(seed, result.transform)
                metrics.update(translation_delta_m=delta_translation, rotation_delta_rad=delta_rotation)
            self.pose_pub.publish(pose)
            self.tf.sendTransform(tf)
            self.axes_pub.publish(axes)
            if not self.visible or time.monotonic() - self.last_overlay >= self.debug_period:
                reference = self.registration.reference_points
                overlay = reference @ result.transform[:3, :3].T + result.transform[:3, 3]
                self.overlay_pub.publish(point_cloud2.create_cloud_xyz32(msg.header, overlay.astype(np.float32)))
                self.last_overlay = time.monotonic()
            self.visible, self.accepted_stamp = True, stamp
            if self.tracking_enabled:
                self.tracking_seed.accept(result.transform, stamp, self.get_clock().now().nanoseconds)
            self.status('icp_accepted', **metrics, pose_total_ms=(time.perf_counter() - start) * 1000,
                        receipt_to_pose_ms=self.age(msg.header) * 1000)
        except Exception as exc:
            self.clear_visuals()
            self.status(f'icp_error:{exc}', 2)
            if self.config.icp_backend in ('cuda', 'cuda_legacy') and isinstance(exc, RuntimeError):
                raise


def main(args=None):
    run(PoseNode, args)
