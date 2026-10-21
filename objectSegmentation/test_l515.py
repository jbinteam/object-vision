"""Live L515 test of the trained YOLO segmentation model. S saves; Q exits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np

from l515_protocol import read_message

ROOT = Path(__file__).resolve().parent
WINDOW = "L515 - emergency_stop segmentation"


class CameraStream:
    def __init__(self, args):
        self.args = args
        self.process = None

    def __enter__(self):
        command = [str(self.args.camera_python), "-E", str(ROOT / "l515_camera_worker.py"),
                   "--width", str(self.args.width), "--height", str(self.args.height), "--fps", str(self.args.fps)]
        if self.args.serial:
            command += ["--serial", self.args.serial]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        try:
            self.info, _ = read_message(self.process.stdout, timeout=20)
            if self.info["type"] != "ready":
                raise RuntimeError(self.info.get("message", "Camera startup failed"))
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        self.process.stdin.write(b"F")
        self.process.stdin.flush()
        metadata, payload = read_message(self.process.stdout)
        if metadata["type"] != "frame":
            raise RuntimeError(metadata.get("message", "Invalid camera response"))
        shape = (self.args.height, self.args.width, 3)
        if metadata["shape"] != list(shape) or len(payload) != math.prod(shape):
            raise ValueError("Camera frame dimensions or payload are invalid")
        return np.frombuffer(payload, np.uint8).reshape(shape), metadata

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(b"Q")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
        finally:
            self.process.stdin.close()
            self.process.stdout.close()

    def __exit__(self, *exc):
        self.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/segment/emergency_stop_backgrounds/weights/best.pt")
    parser.add_argument("--camera-python", type=Path, default=ROOT / ".venv-camera/bin/python")
    parser.add_argument("--serial")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto", help="auto, cpu, or GPU index")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N predictions; 0 means unlimited")
    parser.add_argument("--save-every", type=int, default=0, help="Save a snapshot every N predictions; 0 means manual")
    parser.add_argument("--output", type=Path, default=ROOT / "output/l515_test")
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps, args.imgsz) <= 0 or min(args.max_frames, args.save_every) < 0:
        parser.error("Dimensions/FPS must be positive and frame limits nonnegative")
    if not (0 < args.conf < 1 and 0 < args.iou <= 1):
        parser.error("Require 0 < conf < 1 and 0 < iou <= 1")
    if args.device not in ("auto", "cpu") and not args.device.isdigit():
        parser.error("--device must be auto, cpu, or a GPU index")
    if not args.camera_python.is_file():
        parser.error(f"Camera Python missing: {args.camera_python}")
    return args


def save_snapshot(session, index, frame, annotated, result, record):
    folder = session / f"snapshot_{index:06d}"
    folder.mkdir(exist_ok=False)
    products = {"color.png": frame, "prediction.png": annotated}
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for number, mask in enumerate(masks):
            if mask.shape != frame.shape[:2]:
                raise ValueError("Predicted mask is not at original camera resolution")
            products[f"mask_{number:03d}.png"] = (mask > 0.5).astype(np.uint8) * 255
    for name, image in products.items():
        if not cv2.imwrite(str(folder / name), image):
            raise OSError(f"Cannot save {folder / name}")
    (folder / "predictions.json").write_text(json.dumps({**record,
        "coordinates": "normalized to original frame width/height",
        "instances": result.summary(normalize=True, decimals=6)}, indent=2) + "\n")
    print(f"Saved: {folder}", flush=True)


def run(args):
    if args.list:
        return subprocess.call([str(args.camera_python), "-E", str(ROOT / "l515_camera_worker.py"), "--list"])
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights missing: {args.weights}; choose --weights /path/to/best.pt")
    if not args.no_preview and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No display found. Use --no-preview --max-frames 100 --save-every 20.")
    config_dir = ROOT / ".codex/ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    import torch
    from ultralytics import YOLO, settings
    settings.update({key: False for key in ("sync", "hub", "wandb", "comet", "clearml", "mlflow") if key in settings})
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = YOLO(str(args.weights.resolve()))
    if model.task != "segment":
        raise ValueError("The selected checkpoint is not a segmentation model")
    prediction_args = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device,
                           retina_masks=True, max_det=20, verbose=False, save=False)
    # Warm model kernels before measuring the live loop.
    model.predict(np.zeros((args.height, args.width, 3), np.uint8), **prediction_args)
    print(f"Weights: {args.weights.resolve()}\nClasses: {model.names}\nDevice: {device}", flush=True)
    count = snapshots = frames_with_detections = 0
    session = None
    opened = False
    status = "completed"
    started = time.perf_counter()
    try:
        with CameraStream(args) as camera:
            folder = args.output.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8])
            folder.mkdir(parents=True, exist_ok=False)
            session = folder
            (session / "session.json").write_text(json.dumps({
                "weights": str(args.weights.resolve()), "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                "camera": camera.info, "class_names": model.names, "prediction": prediction_args,
                "purpose": "live prediction test; confidence scores are not accuracy measurements",
            }, indent=2) + "\n")
            print(f"Results: {session}\nS/Space: snapshot | M: masks | +/-: confidence | Q/Esc: quit", flush=True)
            if not args.no_preview:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, 1280, 720)
                opened = True
            show_masks = True
            average_fps = 0.0
            started = time.perf_counter()
            with (session / "frames.jsonl").open("x") as log:
                while True:
                    tick = time.perf_counter()
                    frame, metadata = camera.read()
                    result = model.predict(frame, **prediction_args)[0]
                    count += 1
                    detections = len(result.boxes)
                    frames_with_detections += int(detections > 0)
                    record = {"index": count, "captured_utc": datetime.now(timezone.utc).isoformat(),
                              "camera_frame": metadata["frame_number"], "camera_timestamp_ms": metadata["camera_timestamp_ms"],
                              "confidence_threshold": prediction_args["conf"], "detections": detections,
                              "confidence_scores": result.boxes.conf.cpu().tolist(), "speed_ms": result.speed}
                    annotated = result.plot(masks=show_masks, boxes=True, labels=True, conf=True, line_width=2)
                    hz = 1 / max(time.perf_counter()-tick, 1e-6)
                    average_fps = hz if not average_fps else 0.9*average_fps+0.1*hz
                    text = f"{detections} objects | {average_fps:.1f} processing FPS | infer {result.speed['inference']:.1f} ms | conf {prediction_args['conf']:.2f}"
                    cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 1000), 44), (25, 25, 25), -1)
                    cv2.putText(annotated, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                    key = -1
                    if opened:
                        cv2.imshow(WINDOW, annotated)
                        key = cv2.waitKey(1) & 0xFF
                    if (args.save_every and count % args.save_every == 0) or key in (ord("s"), ord("S"), 32):
                        save_snapshot(session, count, frame, annotated, result, record)
                        snapshots += 1
                    log.write(json.dumps(record)+"\n")
                    log.flush()
                    if count == 1 or count % 30 == 0:
                        print(f"Frame {count}: {text}", flush=True)
                    if key in (ord("m"), ord("M")):
                        show_masks = not show_masks
                    if key in (ord("+"), ord("=")):
                        prediction_args["conf"] = min(.95, round(prediction_args["conf"]+.05, 2))
                    if key in (ord("-"), ord("_")):
                        prediction_args["conf"] = max(.05, round(prediction_args["conf"]-.05, 2))
                    if key in (ord("q"), ord("Q"), 27) or (opened and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                        break
                    if args.max_frames and count >= args.max_frames:
                        break
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "error"
        raise
    finally:
        if opened:
            cv2.destroyAllWindows()
        if session is not None:
            summary = {"status": status, "frames": count, "frames_with_detections": frames_with_detections,
                       "snapshots": snapshots, "elapsed_seconds": time.perf_counter()-started}
            (session / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
            print(f"Finished: {count} frames, {snapshots} snapshots. Results: {session}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (RuntimeError, OSError, ValueError, EOFError, cv2.error) as error:
        raise SystemExit(f"Live test error: {error}")
