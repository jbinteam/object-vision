#!/usr/bin/env python3
"""Runs with the scanner's camera environment and selected SDK; no ROS imports."""
import argparse
import contextlib
import os
import sys
from protocol import write_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--camera-backend', choices=('orbbec', 'realsense', 'auto'), default='orbbec')
    parser.add_argument('--serial', default='')
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--minimum-depth', type=float, default=0.15)
    parser.add_argument('--maximum-depth', type=float, default=1.5)
    args = parser.parse_args()
    if args.downscale < 1 or not 0 < args.minimum_depth < args.maximum_depth <= 65.535:
        parser.error('Invalid downscale or depth bounds')
    output = sys.stdout.buffer
    # Native libraries and Python camera logs must never corrupt the binary pipe.
    output = os.fdopen(os.dup(output.fileno()), 'wb')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.path.insert(0, args.project_root)
    with contextlib.redirect_stdout(sys.stderr):
        from camera import open_camera
        camera = open_camera(args.camera_backend, args.serial or None)
        try:
            while True:
                frame = camera.read(args.minimum_depth, args.maximum_depth, 1.0, args.downscale)
                if frame is not None:
                    write_frame(output, frame.color, frame.depth, frame.intrinsic, frame.timestamp_ns)
        finally:
            camera.close()


if __name__ == '__main__':
    main()
