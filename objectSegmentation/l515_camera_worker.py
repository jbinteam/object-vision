"""Camera-only worker. Started automatically by test_l515.py using .venv-camera."""
import argparse
import sys
import time

import numpy as np
import pyrealsense2 as rs

from capture_l515 import color_modes
from l515_protocol import write_message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    pipeline = None
    started = False
    try:
        context = rs.context()
        devices = list(context.query_devices())
        if args.list:
            if not devices:
                print("No RealSense cameras found. Connect the L515 via USB 3.")
            for device in devices:
                print(f"{device.get_info(rs.camera_info.name)} | serial {device.get_info(rs.camera_info.serial_number)}")
                print("Color modes: " + ", ".join(f"{w}x{h}@{fps}" for w,h,fps in color_modes(device, rs)))
            return
        devices = [d for d in devices if "515" in d.get_info(rs.camera_info.name)
                   and (not args.serial or d.get_info(rs.camera_info.serial_number) == args.serial)]
        if len(devices) != 1:
            raise RuntimeError("Expected one matching L515. Connect the camera, run --list, and use --serial if needed.")
        device = devices[0]
        modes = color_modes(device, rs)
        if (args.width, args.height, args.fps) not in modes:
            raise RuntimeError(f"Unsupported color mode; supported modes: {modes}")
        serial = device.get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        pipeline = rs.pipeline(context)
        pipeline.start(config)
        started = True
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            pipeline.wait_for_frames(5000)
        write_message(sys.stdout.buffer, {"type": "ready", "camera": device.get_info(rs.camera_info.name),
                      "serial": serial, "width": args.width, "height": args.height, "fps": args.fps})
        while True:
            command = sys.stdin.buffer.read(1)
            if command in (b"", b"Q"):
                break
            if command != b"F":
                raise RuntimeError("Invalid camera request")
            # RealSense drops old unread frames. Only one frame is transferred per request.
            color = pipeline.wait_for_frames(5000).get_color_frame()
            if not color:
                raise RuntimeError("No color frame received")
            frame = np.asanyarray(color.get_data())
            if frame.shape != (args.height, args.width, 3):
                raise RuntimeError(f"Unexpected color frame shape: {frame.shape}")
            write_message(sys.stdout.buffer, {"type": "frame", "shape": list(frame.shape),
                          "camera_timestamp_ms": color.get_timestamp(), "frame_number": color.get_frame_number()},
                          frame.tobytes())
    except BrokenPipeError:
        pass
    except Exception as error:
        print(f"L515: {error}", file=sys.stderr, flush=True)
        if not args.list:
            try:
                write_message(sys.stdout.buffer, {"type": "error", "message": str(error)})
            except BrokenPipeError:
                pass
        raise SystemExit(1)
    finally:
        if started:
            pipeline.stop()


if __name__ == "__main__":
    main()
