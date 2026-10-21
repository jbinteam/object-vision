"""Synchronized mask + aligned depth + rectified calibration -> object cloud."""
import time
import numpy as np
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from .geometry import CloudConfig, backproject, preprocess, transform_points
from .cloud_messages import create_colored_cloud
from .ros_support import MovingAverageMetrics, PipelineNode, run, stamp_ns


class CloudNode(PipelineNode):
    def __init__(self):
        super().__init__('object_cloud')
        self.config = CloudConfig(**{name: self.param(name, default)
                                    for name, default in vars(CloudConfig()).items()})
        self.frame = self.param('camera_frame', 'camera_color_optical_frame')
        if not self.frame or self.frame.startswith('/'):
            raise ValueError('camera_frame must be a nonempty relative frame name')
        self.max_age = self.positive('maximum_data_age', 0.75)
        queue = int(self.param('sync_queue_size', 20))
        slop = float(self.param('sync_slop', 0.0))
        if queue < 1 or not np.isfinite(slop) or slop < 0 or slop > self.max_age:
            raise ValueError('Invalid sync queue size or slop')
        self.bridge = CvBridge()
        self.timings = MovingAverageMetrics()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(PointCloud2, '/object/cloud', self.image_qos(5))
        self.subscribers = [message_filters.Subscriber(self, kind, topic, qos_profile=self.image_qos(queue))
                            for kind, topic in [(Image, '/object/mask'), (Image, '/camera/depth/image_raw'),
                                                (CameraInfo, '/camera/camera_info'),
                                                (Image, '/camera/color/image_raw')]]
        if slop == 0:
            self.synchronizer = message_filters.TimeSynchronizer(self.subscribers, queue)
        else:
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(self.subscribers, queue, slop)
        self.synchronizer.registerCallback(self.extract)
        self.last_stamp = -1
        self.create_timer(0.25, self.check_timeout)
        self.status('cloud_ready')

    def check_timeout(self):
        if time.monotonic() - self.last_input > self.max_age:
            self.status('no_recent_synchronized_input', 1)

    def extract(self, mask_msg, depth_msg, info, color_msg):
        start = time.perf_counter()
        stamp = stamp_ns(depth_msg.header.stamp)
        inputs = (mask_msg, depth_msg, info, color_msg)
        if stamp <= self.last_stamp or any(not 0 <= self.age(m.header) <= self.max_age for m in inputs):
            self.status('stale_or_out_of_order_input', 1)
            return
        self.last_stamp, self.last_input = stamp, time.monotonic()
        header = Header(stamp=depth_msg.header.stamp, frame_id=self.frame)
        try:
            frames = {m.header.frame_id for m in inputs}
            if len(frames) != 1 or not depth_msg.header.frame_id:
                raise ValueError('Mask, RGB, aligned depth, and camera_info must share the optical frame')
            if mask_msg.encoding != 'mono8':
                raise ValueError('Mask must use mono8 encoding')
            if (info.width, info.height) != (depth_msg.width, depth_msg.height):
                raise ValueError('CameraInfo dimensions must match aligned depth')
            if not np.isfinite(info.d).all() or not np.allclose(info.d, 0):
                raise ValueError('Rectified camera_info with zero distortion is required')
            if info.binning_x not in (0, 1) or info.binning_y not in (0, 1) or info.roi.x_offset or info.roi.y_offset:
                raise ValueError('Provide intrinsics already adjusted for binning/ROI')
            if not (np.allclose(info.r, np.eye(3).ravel()) or np.allclose(info.r, 0)):
                raise ValueError('Provide calibration in the rectified optical frame')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
            rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
            points, colors = backproject(depth, mask, np.asarray(info.k).reshape(3, 3),
                                         self.config, depth_msg.encoding, rgb=rgb)
            extracted = time.perf_counter()
            input_points = len(points)
            points, colors = preprocess(points, self.config, colors)
            processed = time.perf_counter()
            if len(points) < self.config.minimum_points:
                points = np.empty((0, 3), dtype=np.float32)
                colors = np.empty((0, 3), dtype=np.uint8)
                reason = 'insufficient_object_points'
            else:
                reason = 'object_cloud_valid'
                if depth_msg.header.frame_id != self.frame:
                    tf = self.tf_buffer.lookup_transform(self.frame, depth_msg.header.frame_id,
                                                         Time.from_msg(depth_msg.header.stamp))
                    t, q = tf.transform.translation, tf.transform.rotation
                    points = transform_points(points, [t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
            if self.age(depth_msg.header) > self.max_age:
                self.status('cloud_result_expired', 1)
                return
            self.publisher.publish(create_colored_cloud(header, points, colors))
            extraction_ms = (extracted - start) * 1000
            preprocessing_ms = (processed - extracted) * 1000
            total_ms = (time.perf_counter() - start) * 1000
            averages = self.timings.update(cloud_extraction_ms=extraction_ms,
                                           preprocessing_ms=preprocessing_ms,
                                           cloud_total_ms=total_ms)
            self.status(reason, 0 if len(points) else 1, input_points=input_points, points=len(points),
                        cloud_extraction_ms=extraction_ms,
                        preprocessing_ms=preprocessing_ms,
                        cloud_total_ms=total_ms, **averages,
                        receipt_to_cloud_ms=self.age(depth_msg.header) * 1000)
        except Exception as exc:
            self.publisher.publish(create_colored_cloud(header, np.empty((0, 3), np.float32),
                                                        np.empty((0, 3), np.uint8)))
            self.status(f'cloud_error: {exc}', 2)


def main(args=None):
    run(CloudNode, args)
