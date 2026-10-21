"""One RGB frame -> paired object mask and ordered YOLO pose keypoints."""
import time
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

from .keypoint_messages import create_keypoints
from .ros_support import MovingAverageMetrics, PipelineNode, run, stamp_ns
from .segmentation import InstanceSelector, YoloPoseEstimator, YoloSegmenter, associate_pose


def draw_detection_box(rgb, detection):
    """Return the RGB frame with the selected segmentation box overlaid."""
    image = rgb.copy()
    if detection is None:
        return image
    box = np.asarray(detection.box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return image
    height, width = image.shape[:2]
    x0, y0, x1, y1 = np.rint(box).astype(int)
    x0, x1 = np.clip([x0, x1], 0, width - 1)
    y0, y1 = np.clip([y0, y1], 0, height - 1)
    if x1 <= x0 or y1 <= y0:
        return image
    color = (0, 255, 0)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    label = f'Object {detection.confidence:.2f}'
    text_y = y0 - 7 if y0 >= 20 else min(height - 5, y0 + 18)
    cv2.putText(image, label, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                .55, color, 2, cv2.LINE_AA)
    return image


class ObservationNode(PipelineNode):
    def __init__(self):
        super().__init__('yolo_observation')
        self.max_age = self.positive('maximum_data_age', .75)
        self.expected_keypoints = int(self.param('expected_keypoints', 8))
        self.minimum_pair_iou = self.positive('pose_segmentation_minimum_iou', .1)
        self.pose_only_for_reinitialization = self.param('pose_only_for_reinitialization', True)
        self.tracking_state = 'INITIALIZATION'
        self.selector = InstanceSelector(self.param('association_minimum_iou', .1),
                                         self.param('association_reset_after_misses', 5))
        common = (self.param('yolo_device', 'cpu'), self.param('yolo_image_size', 640),
                  self.param('cpu_threads', 2), self.param('yolo_half', False))
        self.segmenter = YoloSegmenter(
            self.param('segmentation_model_path', ''),
            self.param('segmentation_target_class', 'my_object'),
            self.param('segmentation_confidence_threshold', .5), *common)
        self.pose = YoloPoseEstimator(
            self.param('pose_model_path', ''), self.param('pose_target_class', 'my_object'),
            self.param('pose_detection_confidence_threshold', .25), *common,
            expected_keypoints=self.expected_keypoints)
        self.bridge = CvBridge()
        self.timings = MovingAverageMetrics()
        self.mask_pub = self.create_publisher(Image, '/object/mask', self.image_qos(5))
        self.detection_image_pub = self.create_publisher(
            Image, '/object_pose/detection_image', self.image_qos(2))
        self.keypoint_pub = self.create_publisher(PointCloud2, '/object/keypoints', self.image_qos(5))
        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.image, self.image_qos())
        self.state_subscription = self.create_subscription(
            String, '/object_pose/tracking_state', self.tracking_update, 10)
        self.last_stamp = -1
        self.status('observation_ready', device=self.segmenter.device,
                    keypoints=self.expected_keypoints)

    def tracking_update(self, message):
        if message.data in ('INITIALIZATION', 'TRACKING', 'LOST'):
            self.tracking_state = message.data

    def image(self, msg):
        started = time.perf_counter()
        stamp = stamp_ns(msg.header.stamp)
        if stamp <= self.last_stamp or not 0 <= self.age(msg.header) <= self.max_age:
            self.status('stale_or_out_of_order_color', 1)
            return
        self.last_stamp, self.last_input = stamp, time.monotonic()
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            segmentation_start = time.perf_counter()
            segmentation, reason = self.selector.select(self.segmenter.predict(rgb))
            segmentation_ms = (time.perf_counter() - segmentation_start) * 1000
            pose_required = (not self.pose_only_for_reinitialization
                             or self.tracking_state != 'TRACKING')
            pose_start = time.perf_counter()
            pose_ran = pose_required and segmentation is not None
            poses = self.pose.predict(rgb) if pose_ran else []
            pose_ms = (time.perf_counter() - pose_start) * 1000
            pose = associate_pose(segmentation, poses, self.minimum_pair_iou)
            mask = (np.zeros(rgb.shape[:2], np.uint8) if segmentation is None
                    else segmentation.mask.astype(np.uint8) * 255)
            keypoints = (np.full((self.expected_keypoints, 4), [np.nan, np.nan, 0., 0.], np.float32)
                         if pose is None else pose.keypoints)
            if self.age(msg.header) > self.max_age:
                self.selector.reset()
                self.status('observation_result_expired', 1)
                return
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)
            if self.detection_image_pub.get_subscription_count():
                annotated = self.bridge.cv2_to_imgmsg(
                    draw_detection_box(rgb, segmentation), encoding='rgb8')
                annotated.header = msg.header
                self.detection_image_pub.publish(annotated)
            self.keypoint_pub.publish(create_keypoints(msg.header, keypoints))
            pose_ok = pose is not None or not pose_required
            pose_status = reason if pose_ok else reason + ':pose_unmatched'
            total_ms = (time.perf_counter() - started) * 1000
            averages = self.timings.update(segmentation_ms=segmentation_ms,
                                           pose_inference_ms=pose_ms,
                                           observation_ms=total_ms)
            self.status(pose_status, 0 if pose_ok else 1,
                        tracking_state=self.tracking_state, pose_inference_required=pose_required,
                        pose_inference_ran=pose_ran,
                        mask_pixels=int(np.count_nonzero(mask)),
                        valid_keypoints=int(np.count_nonzero(keypoints[:, 3])),
                        segmentation_ms=segmentation_ms, pose_inference_ms=pose_ms,
                        observation_ms=total_ms, **averages)
        except Exception as exc:
            self.selector.reset()
            self.status(f'observation_error:{exc}', 2)


def main(args=None):
    run(ObservationNode, args)
