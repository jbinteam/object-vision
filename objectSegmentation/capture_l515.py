"""Capture L515 color PNGs for BiRefNet: Space/S saves, R toggles timed capture."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WINDOW = "L515 dataset capture"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "input" / "l515",
                        help="Parent directory; each run creates a new session folder")
    parser.add_argument("--serial", help="Choose an L515 by serial number")
    parser.add_argument("--list", action="store_true", help="List cameras and available BGR color modes")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between timed captures")
    parser.add_argument("--auto", action="store_true", help="Start timed capture after warmup")
    parser.add_argument("--count", type=int, default=0, help="Stop after N saved frames; 0 means unlimited")
    parser.add_argument("--warmup", type=float, default=2.0, help="Seconds for auto exposure to settle")
    parser.add_argument("--no-preview", action="store_true", help="Headless capture; requires --auto")
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps) <= 0 or args.count < 0:
        parser.error("Dimensions and FPS must be positive; count must be nonnegative")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and greater than zero")
    if not math.isfinite(args.warmup) or args.warmup < 0:
        parser.error("--warmup must be finite and nonnegative")
    if args.no_preview and not args.auto and not args.list:
        parser.error("--no-preview requires --auto")
    return args


def color_modes(device, rs):
    modes = set()
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8:
                video = profile.as_video_stream_profile()
                modes.add((video.width(), video.height(), video.fps()))
    return sorted(modes)


def save_frame(session, log, frame, index, timestamp_ms, frame_number):
    """Write the unmodified BGR camera frame and an immediately flushed log row."""
    captured = datetime.now(timezone.utc)
    filename = f"frame_{index:06d}_{captured.strftime('%Y%m%dT%H%M%S_%fZ')}.png"
    success, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise OSError("PNG encoding failed")
    with (session / filename).open("xb") as image_file:
        image_file.write(encoded.tobytes())
    log.write(json.dumps({"file": filename, "captured_at_utc": captured.isoformat(),
                          "camera_timestamp_ms": timestamp_ms, "frame_number": frame_number,
                          "width": frame.shape[1], "height": frame.shape[0]}) + "\n")
    log.flush()
    print(f"Saved {session / filename}", flush=True)


def capture(args, rs):
    context = rs.context()
    devices = list(context.query_devices())
    if args.list:
        if not devices:
            print("No RealSense devices found. Connect the L515 with a USB 3 data cable.")
        for device in devices:
            print(f"{device.get_info(rs.camera_info.name)} | serial {device.get_info(rs.camera_info.serial_number)}")
            print("  Color BGR8 modes: " + ", ".join(f"{w}x{h}@{fps}" for w, h, fps in color_modes(device, rs)))
        return
    devices = [d for d in devices if "515" in d.get_info(rs.camera_info.name)]
    if args.serial:
        devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == args.serial]
    if not devices:
        raise RuntimeError("No matching L515 found. Connect it via USB 3, close other camera apps, "
                           "then run --list. On Linux, check RealSense udev permissions.")
    if len(devices) > 1:
        raise RuntimeError("Multiple L515 cameras found. Run --list, then select --serial SERIAL.")
    device = devices[0]
    requested = (args.width, args.height, args.fps)
    modes = color_modes(device, rs)
    if requested not in modes:
        raise RuntimeError(f"Color mode {requested} is unavailable. Available BGR8 modes: {modes}")
    if not args.no_preview and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No graphical display found. Use --no-preview --auto --count N.")
    serial = device.get_info(rs.camera_info.serial_number)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    started = False
    session = None
    saved = 0
    try:
        pipeline.start(config)
        started = True
        session = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8])
        session.mkdir(parents=True, exist_ok=False)
        metadata = {"camera": device.get_info(rs.camera_info.name), "serial": serial,
                    "firmware": device.get_info(rs.camera_info.firmware_version),
                    "width": args.width, "height": args.height, "fps": args.fps,
                    "format": "color PNG", "interval_seconds": args.interval,
                    "warmup_seconds": args.warmup, "sdk": "2.54.2.5684"}
        (session / "session.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if not args.no_preview:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 960, 540)
        print(f"Session: {session}\nSpace/S: save | R: toggle timed capture | Q/Esc: quit", flush=True)
        ready_at = time.monotonic() + args.warmup
        next_save = ready_at
        recording = args.auto
        with (session / "frames.jsonl").open("x") as log:
            while True:
                frames = pipeline.wait_for_frames(5000)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError("No color frame received from the L515.")
                frame = np.asanyarray(color.get_data())
                if frame.shape != (args.height, args.width, 3):
                    raise RuntimeError(f"Unexpected color frame shape: {frame.shape}")
                now = time.monotonic()
                warming = now < ready_at
                key = -1
                if not args.no_preview:
                    # Overlay only the preview; saved images retain untouched camera pixels.
                    scale = min(1.0, 1280 / args.width)
                    preview = cv2.resize(frame, (int(args.width * scale), int(args.height * scale)))
                    state = "WARMUP" if warming else ("TIMED CAPTURE" if recording else "MANUAL")
                    cv2.putText(preview, f"{state} | saved {saved} | S/Space save | R timed | Q quit",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.imshow(WINDOW, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key in (ord("r"), ord("R")):
                        recording = not recording
                        next_save = max(ready_at, now)
                manual = key in (ord("s"), ord("S"), ord(" "))
                if not warming and (manual or (recording and now >= next_save)):
                    save_frame(session, log, frame, saved + 1, color.get_timestamp(), color.get_frame_number())
                    saved += 1
                    next_save = time.monotonic() + args.interval
                    if args.count and saved >= args.count:
                        break
    finally:
        if started:
            pipeline.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()
        if session is not None:
            print(f"Saved {saved} image(s) in {session}", flush=True)


def main():
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise SystemExit("RealSense SDK missing. Run with .venv-camera/bin/python -E capture_l515.py")
    try:
        capture(args, rs)
    except KeyboardInterrupt:
        print("Capture stopped.")
    except (RuntimeError, OSError, cv2.error) as error:
        raise SystemExit(f"Capture error: {error}")


if __name__ == "__main__":
    main()
