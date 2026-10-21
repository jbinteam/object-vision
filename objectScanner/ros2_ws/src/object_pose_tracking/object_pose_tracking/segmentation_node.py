"""Color -> same-timestamp binary mask; one latest queued image."""
import time
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from .segmentation import InstanceSelector, YoloSegmenter
from .ros_support import PipelineNode, run, stamp_ns


class SegmentationNode(PipelineNode):
    def __init__(self):
        super().__init__('yolo_segmentation')
        self.max_age = self.positive('maximum_data_age', 0.75)
        self.selector = InstanceSelector(self.param('association_minimum_iou', 0.1),
                                         self.param('association_reset_after_misses', 5))
        self.backend = YoloSegmenter(self.param('yolo_model_path', ''),
                                     self.param('target_object_class', 'emergency_stop'),
                                     self.param('yolo_confidence_threshold', 0.5),
                                     self.param('yolo_device', 'cpu'),
                                     self.param('yolo_image_size', 640),
                                     self.param('cpu_threads', 2),
                                     self.param('yolo_half', False))
        # Warm up once so the first real observation does not pay setup latency.
        self.backend.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/object/mask', self.image_qos(5))
        self.subscription = self.create_subscription(Image, '/camera/color/image_raw', self.image, self.image_qos())
        self.last_stamp = -1
        self.last_input = time.monotonic()
        self.create_timer(0.25, self.check_timeout)
        self.status('segmentation_ready', device=self.backend.device, half=self.backend.half)

    def check_timeout(self):
        if time.monotonic() - self.last_input > self.max_age:
            self.selector.reset()
            self.status('no_recent_color', 1)

    def image(self, msg):
        start = time.perf_counter()
        stamp = stamp_ns(msg.header.stamp)
        if stamp <= self.last_stamp or not 0 <= self.age(msg.header) <= self.max_age:
            self.status('stale_or_out_of_order_color', 1)
            return
        if time.monotonic() - self.last_input > self.max_age:
            self.selector.reset()
        self.last_input, self.last_stamp = time.monotonic(), stamp
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            detections = self.backend.predict(rgb)
            selected, reason = self.selector.select(detections)
            mask = np.zeros(rgb.shape[:2], np.uint8) if selected is None else selected.mask.astype(np.uint8) * 255
            if self.age(msg.header) > self.max_age:
                self.selector.reset()
                self.status('inference_result_expired', 1)
                return
            output = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            output.header = msg.header  # acquisition stamp, never inference-completion time
            self.publisher.publish(output)
            self.status(reason, 1 if selected is None else 0,
                        device=self.backend.device, half=self.backend.half,
                        segmentation_ms=(time.perf_counter() - start) * 1000,
                        receipt_to_mask_ms=self.age(msg.header) * 1000,
                        mask_pixels=int(np.count_nonzero(mask)), detections=len(detections))
        except Exception as exc:
            self.selector.reset()
            self.status(f'segmentation_error: {exc}', 2)
            if self.backend.device != 'cpu' and isinstance(exc, RuntimeError):
                raise  # Stop the launch group rather than retry a faulted CUDA context.
            # Publish an empty mask so downstream consumers clear this observation.
            output = self.bridge.cv2_to_imgmsg(np.zeros((msg.height, msg.width), np.uint8), encoding='mono8')
            output.header = msg.header
            self.publisher.publish(output)


def main(args=None):
    run(SegmentationNode, args)
