#!/usr/bin/env python3
"""Publish a mono8 mask for NVIDIA's one-frame mustard-bottle sample.

This is only a smoke-test adapter for the official rosbag. The live pipeline
will publish masks from the objectSegmentation YOLO model instead.
"""

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class SampleYellowMaskPublisher(Node):
    def __init__(self) -> None:
        super().__init__('sample_yellow_mask_publisher')
        self.declare_parameter('input_topic', '/rgb/image_rect_color')
        self.declare_parameter('output_topic', '/segmentation')
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self._bridge = CvBridge()
        # FoundationPose requests reliable delivery for its segmentation input.
        # Keep the camera subscription on sensor-data QoS, but publish the mask
        # reliably so DDS will connect the endpoints.
        self._publisher = self.create_publisher(Image, output_topic, 10)
        self._subscription = self.create_subscription(
            Image, input_topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f'Publishing sample masks: {input_topic} -> {output_topic}')

    def _on_image(self, message: Image) -> None:
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(
            hsv,
            np.array((16, 70, 60), dtype=np.uint8),
            np.array((42, 255, 255), dtype=np.uint8),
        )
        yellow = cv2.morphologyEx(
            yellow, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
        # The official frame places the mustard bottle in this left-table ROI.
        # Limiting the smoke test avoids selecting similarly colored cartons at
        # the bottom edge of the recorded frame.
        roi = np.zeros_like(yellow)
        roi[220:410, 60:230] = 255
        yellow = cv2.bitwise_and(yellow, roi)
        contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100.0:
                cv2.fillConvexPoly(mask, cv2.convexHull(largest), 255)
        output = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
        output.header = message.header
        self._publisher.publish(output)


def main() -> None:
    rclpy.init()
    node = SampleYellowMaskPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
