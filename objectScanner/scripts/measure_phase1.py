#!/usr/bin/env python3
"""Observe an already running Phase 1 pipeline and report measured throughput.

Run using .venv-ros/bin/python after sourcing ros2_ws/install/setup.bash.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs_py import point_cloud2
from object_pose_tracking.ros_support import sensor_qos, stamp_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--startup-timeout', type=float, default=35)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--best-effort', action='store_true')
    args = parser.parse_args()
    if not 0 < args.seconds <= 600 or not 0 < args.startup_timeout <= 120:
        parser.error('seconds must be in (0,600], startup-timeout in (0,120]')
    rclpy.init()
    node = Node('phase1_measurement')
    counts, states = Counter(), Counter()
    metrics, stamps = defaultdict(list), defaultdict(set)
    examples = []
    first_data = None

    def observe(msg, topic):
        nonlocal first_data
        if first_data is None:
            first_data = time.monotonic()
        counts[topic] += 1
        stamps[topic].add(stamp_ns(msg.header.stamp))
        if topic == 'cloud':
            xyz = point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z'])
            counts['valid_cloud'] += int(len(xyz) > 0)
            if len(xyz) and len(examples) < 2:
                examples.append(dict(points=len(xyz), frame=msg.header.frame_id,
                                     minimum=xyz.min(axis=0).tolist(), maximum=xyz.max(axis=0).tolist(),
                                     finite=bool(np.isfinite(xyz).all())))

    def diagnostic(msg):
        for status in msg.status:
            states[status.name + ':' + status.message] += 1
            for item in status.values:
                try:
                    value = float(item.value)
                except ValueError:
                    continue
                if np.isfinite(value):
                    metrics[status.name + '/' + item.key].append(value)

    subscriptions = [node.create_subscription(kind, topic, lambda m, t=name: observe(m, t),
                                               sensor_qos(20, not args.best_effort))
                     for kind, topic, name in [(Image, '/camera/color/image_raw', 'color'),
                                              (Image, '/camera/depth/image_raw', 'depth'),
                                              (Image, '/object/mask', 'mask'),
                                              (PointCloud2, '/object/cloud', 'cloud')]]
    subscriptions.append(node.create_subscription(DiagnosticArray, '/diagnostics', diagnostic, 20))
    started = time.monotonic()
    try:
        while rclpy.ok():
            now = time.monotonic()
            if first_data is not None and now - first_data >= args.seconds:
                break
            if first_data is None and now - started >= args.startup_timeout:
                break
            rclpy.spin_once(node, timeout_sec=.05)
        duration = time.monotonic() - first_data if first_data is not None else 0
        report = dict(duration_s=duration, counts=dict(counts),
                      cloud_hz=counts['cloud'] / duration if duration else 0,
                      valid_cloud_hz=counts['valid_cloud'] / duration if duration else 0,
                      states=dict(states), sample_clouds=examples,
                      matched_mask_depth_stamps=len(stamps['mask'] & stamps['depth']),
                      matched_cloud_depth_stamps=len(stamps['cloud'] & stamps['depth']),
                      metrics={k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)))
                               for k, v in metrics.items()})
        text = json.dumps(report, indent=2) + '\n'
        print(text)
        if args.output:
            args.output.write_text(text)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
