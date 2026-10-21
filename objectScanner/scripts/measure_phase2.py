#!/usr/bin/env python3
"""Observe live ICP output; report accepted and rejected timings separately."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from diagnostic_msgs.msg import DiagnosticArray
from visualization_msgs.msg import MarkerArray, Marker
from object_pose_tracking.ros_support import sensor_qos


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 0 < args.seconds <= 600:parser.error('seconds must be in (0,600]')
    rclpy.init();node=Node('phase2_measurement')
    counts,states,seed_modes=Counter(),Counter(),Counter();metrics=defaultdict(list);samples=[]
    def pose(msg):
        counts['poses']+=1
        if len(samples)<2:
            p,q=msg.pose.position,msg.pose.orientation
            samples.append(dict(frame=msg.header.frame_id,stamp_sec=msg.header.stamp.sec,
                                stamp_nanosec=msg.header.stamp.nanosec,xyz=[p.x,p.y,p.z],
                                quaternion_xyzw=[q.x,q.y,q.z,q.w]))
    def axes(msg):
        if msg.markers:counts['axes_add' if msg.markers[0].action==Marker.ADD else 'axes_delete']+=1
    def diagnostic(msg):
        for status in msg.status:
            if status.name!='icp_pose':continue
            states[status.message]+=1
            category='accepted' if status.message=='icp_accepted' else 'unaccepted'
            for item in status.values:
                if item.key=='initial_guess':seed_modes[category+'/'+item.value]+=1
                try:value=float(item.value)
                except ValueError:continue
                if np.isfinite(value):metrics[category+'/'+item.key].append(value)
    subs=[node.create_subscription(PoseStamped,'/object/pose',pose,sensor_qos(20)),
          node.create_subscription(MarkerArray,'/object/frame_axes',axes,sensor_qos(20)),
          node.create_subscription(DiagnosticArray,'/diagnostics',diagnostic,20)]
    start=time.monotonic()
    try:
        while rclpy.ok() and time.monotonic()-start<args.seconds:
            rclpy.spin_once(node,timeout_sec=.05)
        duration=time.monotonic()-start
        result=dict(duration_s=duration,counts=dict(counts),accepted_pose_hz=counts['poses']/duration,
                    states=dict(states),seed_modes=dict(seed_modes),pose_samples=samples,
                    metrics={k:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in metrics.items()})
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        print(text)
        if args.output:args.output.write_text(text)
    finally:
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
