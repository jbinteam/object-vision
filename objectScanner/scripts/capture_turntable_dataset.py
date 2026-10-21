#!/usr/bin/env python3
"""Manually capture RGB images with AprilTag-derived object center poses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import quote
import webbrowser

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import open_camera  # noqa: E402
from centerpose_dataset import (export_centerpose_dataset,  # noqa: E402
                                export_centerpose_sample, validate_category)
from turntable_dataset import (EDGES, TAG_FAMILIES, calibrate, detect_tags, make_detector, make_label,
                               reconfigure_box, register_visible_tags,
                               relabel_for_reconfigured_box, remove_tags)  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def ids_arg(value):
    try:
        ids = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer tag IDs") from exc
    if len(ids) != len(set(ids)) or not ids:
        raise argparse.ArgumentTypeError("Tag IDs must be unique")
    return ids


def dataset_name_arg(value):
    if (not value or value != value.strip() or value in {".", ".."}
            or "/" in value or "\\" in value):
        raise argparse.ArgumentTypeError(
            "Dataset name must be a single nonempty directory name")
    return value


def category_arg(value):
    try:
        return validate_category(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def arguments(argv=None, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--name", "--dataset-name", dest="dataset_name", type=dataset_name_arg,
        help="Dataset name; saves to datasets/NAME")
    destination.add_argument(
        "--output", type=Path, help="Explicit new or existing dataset session directory")
    parser.add_argument("--centerpose-category", "--category", type=category_arg,
                        help="Object category for exported training data (default: dataset directory name)")
    parser.add_argument("--centerpose-split", "--split",
                        choices=("train", "val", "test"), default="train",
                        help="Training-data split for this capture session (default: train)")
    parser.add_argument("--setup-tag-size-mm", type=float, default=14,
                        help="Temporary box tag edge length, excluding quiet zone (default: 14)")
    parser.add_argument("--permanent-tag-size-mm", type=float, default=28,
                        help="Permanent table tag edge length, excluding quiet zone (default: 28)")
    parser.add_argument("--tag-family", choices=TAG_FAMILIES, default="tagStandard41h12",
                        help="AprilTag family used for every printed tag (default: tagStandard41h12)")
    parser.add_argument("--setup-ids", type=ids_arg, default=(1, 2, 3, 4),
                        help="Four unordered setup tags: three surface corners and one top")
    parser.add_argument("--permanent-ids", type=ids_arg, default=(5, 6, 7, 8, 9, 10),
                        help="All fixed turntable tag IDs; first visible ID defines table frame")
    parser.add_argument("--camera-backend", choices=("orbbec", "realsense", "auto"), default="orbbec",
                        help="RGB-D backend (default: Orbbec Gemini 305)")
    parser.add_argument("--serial", help="Camera serial if more than one is connected")
    parser.add_argument("--depth-min", type=float,
                        help="Near depth cutoff in metres (default: 0.04 for Orbbec, 0.25 otherwise)")
    parser.add_argument("--depth-max", type=float, default=1.5,
                        help="Far depth cutoff in metres (default: 1.5)")
    parser.add_argument("--reference", type=Path,
                        help="Use an existing canonical reference.ply instead of requiring an in-session scan")
    parser.add_argument("--scan-seconds", type=float, default=30,
                        help="TSDF reference-scan duration started by P (default: 30)")
    parser.add_argument("--scan-voxel-mm", type=float, default=3,
                        help="TSDF voxel size for the raw reference scan (default: 3)")
    parser.add_argument("--scan-downscale", type=int, choices=(1, 2, 4), default=1,
                        help="Reference-scan RGB-D downscale (default: 1 for CUDA)")
    parser.add_argument("--scan-stall-timeout", type=float, default=60,
                        help="Seconds allowed for one native scan operation (default: 60)")
    parser.add_argument("--scan-device", choices=("cpu", "cuda", "auto"), default="cuda",
                        help="Reference-scan compute device (default: cuda)")
    parser.add_argument("--reference-editor-port", type=int, default=0,
                        help="Local reference-editor port; 0 chooses an available port")
    parser.add_argument("--no-open-reference-editor", action="store_true",
                        help="Start the editor without opening a browser tab")
    args = parser.parse_args(argv)
    if args.depth_min is None:
        args.depth_min = 0.04 if args.camera_backend == "orbbec" else 0.25
    if args.dataset_name is not None:
        args.output = Path("datasets") / args.dataset_name
    if args.centerpose_category is None:
        try:
            args.centerpose_category = validate_category(args.output.name)
        except ValueError as exc:
            parser.error(f"Cannot derive CenterPose category from output path: {exc}")
    if len(args.setup_ids) != 4 or len(args.permanent_ids) < 2:
        parser.error("Need exactly four setup IDs and at least two permanent IDs")
    if set(args.setup_ids) & set(args.permanent_ids):
        parser.error("Setup and permanent IDs must be distinct")
    if not 5 <= args.setup_tag_size_mm <= 200:
        parser.error("Setup tag size must be between 5 and 200 mm")
    if not 5 <= args.permanent_tag_size_mm <= 200:
        parser.error("Permanent tag size must be between 5 and 200 mm")
    if not 0 < args.depth_min < args.depth_max:
        parser.error("Depth limits must be positive and increasing")
    if args.scan_seconds <= 0 or args.scan_voxel_mm <= 0 or args.scan_stall_timeout <= 0:
        parser.error("Reference scan duration, voxel size, and stall timeout must be positive")
    if not 0 <= args.reference_editor_port <= 65535:
        parser.error("Reference editor port must be between 0 and 65535")
    if args.reference is not None:
        args.reference = args.reference.expanduser().resolve()
        if not args.reference.is_file() or args.reference.suffix.lower() != ".ply":
            parser.error("--reference must be an existing .ply file")
    return args


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def resumable_empty_scaffold(output, category, split):
    """Accept only the empty directory layout this collector creates itself.

    Camera or detector startup can fail after the folders and CenterPose metadata
    are created but before calibration. A later run may safely reuse that exact
    scaffold; any sample, unknown file, symlink, or mismatched export remains a
    hard error.
    """
    output = Path(output)
    pair = Path("centerpose/outf") / f"{category}_{split}"
    allowed_directories = {
        Path("images"), Path("raw_images"), Path("depths"), Path("labels"),
        Path("centerpose"), Path("centerpose/outf"), pair,
        Path("reference_scans"), Path("references"),
    }
    metadata_path = output / "centerpose/export.json"
    allowed_files = {Path("centerpose/export.json")}
    for path in output.rglob("*"):
        relative = path.relative_to(output)
        if path.is_symlink():
            return False
        if path.is_dir():
            if (relative not in allowed_directories
                    and relative.parts[0] not in ("reference_scans", "references")):
                return False
        elif relative not in allowed_files:
            if relative.parts[0] == "reference_scans":
                if path.suffix.lower() not in (".ply", ".json", ".log"):
                    return False
            elif relative.parts[0] == "references":
                if path.name not in ("reference.ply", "frame.json", "README.txt",
                                     "reference_bundle.zip"):
                    return False
            else:
                return False
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if metadata.get("category") != category or metadata.get("split") != split:
            return False
    return True


def apply_reconfiguration(output, old_calibration, new_calibration):
    """Archive old geometry, then atomically replace every pose label and calibration."""
    replacements = []
    for label_path in sorted((output / "labels").glob("*.json")):
        old_label = json.loads(label_path.read_text())
        replacements.append((label_path, relabel_for_reconfigured_box(
            old_label, old_calibration, new_calibration)))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive = output / "reconfiguration_history" / (
        f"{stamp}_revision_{old_calibration.get('revision', 1)}")
    archive_labels = archive / "labels"
    archive_labels.mkdir(parents=True)
    save_json(archive / "calibration.json", old_calibration)
    for label_path, _new_label in replacements:
        shutil.copy2(label_path, archive_labels / label_path.name)
    centerpose_root = output / "centerpose" / "outf"
    if centerpose_root.exists():
        for centerpose_label in centerpose_root.rglob("*.json"):
            relative = centerpose_label.relative_to(centerpose_root)
            destination = archive / "centerpose" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(centerpose_label, destination)

    for label_path, new_label in replacements:
        save_json(label_path, new_label)
    save_json(output / "calibration.json", new_calibration)
    return len(replacements), archive


def draw_box(view, label):
    uv = np.rint(label["box_corners_uv_px"]).astype(int)
    for start, end in EDGES:
        cv2.line(view, tuple(uv[start]), tuple(uv[end]), (0, 220, 255), 2)
    center = tuple(np.rint(label["center_uv_px"]).astype(int))
    cv2.drawMarker(view, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)


def button(view, left, right, name, enabled, top=None):
    height = view.shape[0]
    top = height - 46 if top is None else top
    cv2.rectangle(view, (left, top), (right, top + 38),
                  (50, 130, 55) if enabled else (75, 75, 75), -1)
    cv2.putText(view, name, (left + 10, top + 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def canonical_reference(output, category, explicit=None):
    """Return the one reference allowed to unlock image capture, if available."""
    if explicit is not None:
        return Path(explicit).resolve()
    expected = Path(output) / "references" / category / "reference.ply"
    if expected.is_file():
        return expected.resolve()
    candidates = sorted((Path(output) / "references").glob("*/reference.ply"))
    return candidates[0].resolve() if len(candidates) == 1 else None


def available_local_port(requested=0):
    if requested:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def start_reference_editor(output, category, port=0, open_browser=True):
    """Start the existing crop/axis editor for this capture session."""
    scan_dir = Path(output).resolve() / "reference_scans"
    reference_dir = Path(output).resolve() / "references"
    scan_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    port = available_local_port(port)
    command = [
        sys.executable, str(ROOT / "webui.py"), "--port", str(port),
        "--scan-dir", str(scan_dir), "--reference-dir", str(reference_dir),
        "--object-frame", "box_center", "--fixed-frame-from-scan",
    ]
    process = subprocess.Popen(command)
    url = (f"http://127.0.0.1:{port}/?reference_name={quote(category)}"
           "&frame_id=box_center&lock_frame=1")
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Reference editor stopped during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        process.terminate()
        raise RuntimeError(f"Reference editor did not start on port {port}")
    if open_browser:
        webbrowser.open(url)
    print(f"Reference editor: {url}", flush=True)
    return process, url


def reference_scan_command(args, destination, calibration_path=None):
    command = [
        sys.executable, str(ROOT / "scanner.py"), "--output", str(destination),
        "--seconds", str(args.scan_seconds), "--voxel", str(args.scan_voxel_mm),
        "--depth-min", str(args.depth_min), "--depth-max", str(args.depth_max),
        "--device", args.scan_device, "--downscale", str(args.scan_downscale),
        "--stall-timeout", str(args.scan_stall_timeout), "--no-3d",
        "--camera-backend", args.camera_backend,
    ]
    if args.serial:
        command.extend(("--serial", args.serial))
    if calibration_path is not None:
        command.extend(("--anchor-calibration", str(Path(calibration_path).resolve())))
    return command


def main(argv=None, description=None, workflow_name="CenterPose"):
    args = arguments(argv, description)
    session_path = args.output / "calibration.json"
    if session_path.exists():
        calibration = json.loads(session_path.read_text())
        saved_setup_size = calibration.get("setup_tag_size_m", calibration.get("tag_size_m"))
        saved_permanent_size = calibration.get("permanent_tag_size_m", calibration.get("tag_size_m"))
        mismatches = []
        if set(calibration["setup_ids"]) != set(args.setup_ids):
            mismatches.append(f"setup IDs {calibration['setup_ids']} != {list(args.setup_ids)}")
        if set(calibration["permanent_ids"]) != set(args.permanent_ids):
            mismatches.append(
                f"permanent IDs {calibration['permanent_ids']} != {list(args.permanent_ids)}")
        if saved_setup_size is None:
            mismatches.append("saved setup tag size is missing")
        elif abs(saved_setup_size - args.setup_tag_size_mm / 1000) > 1e-9:
            mismatches.append(
                f"setup tag size {saved_setup_size * 1000:g} mm != {args.setup_tag_size_mm:g} mm")
        if saved_permanent_size is None:
            mismatches.append("saved permanent tag size is missing")
        elif abs(saved_permanent_size - args.permanent_tag_size_mm / 1000) > 1e-9:
            mismatches.append(
                f"permanent tag size {saved_permanent_size * 1000:g} mm != "
                f"{args.permanent_tag_size_mm:g} mm")
        if calibration["tag_family"] != args.tag_family:
            mismatches.append(f"family {calibration['tag_family']} != {args.tag_family}")
        if mismatches:
            details = "\n  - ".join(mismatches)
            raise SystemExit(
                f"Existing calibration does not match this capture setup:\n  - {details}\n"
                "Use a new --output directory or archive the existing calibration.")
    elif args.output.exists() and any(args.output.iterdir()):
        if not resumable_empty_scaffold(
                args.output, args.centerpose_category, args.centerpose_split):
            raise SystemExit(
                "Output directory is nonempty and has no calibration.json. It contains "
                "data other than this collector's empty startup scaffold; use a new name "
                "or inspect/archive it before retrying.")
        calibration = None
    else:
        calibration = None
    detector = make_detector(args.tag_family)
    camera = open_camera(args.camera_backend, args.serial)
    # Reject an old camera session before creating folders or rewriting exports.
    if calibration and (calibration.get("camera_serial") != camera.info["serial"]
                        or calibration.get("camera_backend") not in (None, camera.info.get("backend"))):
        camera.close()
        raise SystemExit("This session was calibrated with a different camera; use a new dataset name")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "images").mkdir(exist_ok=True)
        (args.output / "raw_images").mkdir(exist_ok=True)
        (args.output / "depths").mkdir(exist_ok=True)
        (args.output / "labels").mkdir(exist_ok=True)
        exported, centerpose_dir = export_centerpose_dataset(
            args.output, args.centerpose_category, args.centerpose_split)
    except (ValueError, OSError, KeyError) as exc:
        camera.close()
        raise SystemExit(f"Cannot prepare CenterPose output: {exc}") from exc
    except BaseException:
        camera.close()
        raise

    actions = []
    window = ("Turntable dataset | P optional scan | E inspect scan | "
              "SPACE capture | S set BB | R reconfigure | A add tags | Q quit")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def click(event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONUP:
            actions.append((x, y))

    cv2.setMouseCallback(window, click)
    print(f"Setup IDs are {args.setup_ids}: place any three at rectangular base corners "
          "and the remaining tag on the object top. Roles are sorted automatically.")
    print(f"Setup tags: {args.setup_tag_size_mm:g} mm; permanent table tags: "
          f"{args.permanent_tag_size_mm:g} mm.")
    print("Keep permanent tags fixed. Two need to be visible for Set object BB (S).")
    print("When new tags overlap two mapped tags, click Add tags (A) to map them.")
    print("First set the object BB, remove the setup tags, and capture all image views.")
    print("After capture, build reviewed masks and the calibrated RGB-D reference "
          "with scripts/collect_process/02_generate_masks.sh and 04_build_reference.sh. "
          "P saves an optional moving-camera TSDF scan for diagnostics.")
    print(f"{workflow_name}: {args.centerpose_category}_{args.centerpose_split} at {centerpose_dir} "
          f"({exported} existing samples synchronized).")
    status = "Waiting for tags"
    editor_process = None
    editor_url = None
    try:
        while True:
            frame = camera.read(args.depth_min, args.depth_max, 1.0, 1)
            if frame is None:
                continue
            detections = detect_tags(frame.color, detector)
            view = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
            for tag_id, corners in detections.items():
                center = np.rint(corners.mean(axis=0)).astype(int)
                cv2.polylines(view, [np.rint(corners).astype(np.int32)], True,
                              (0, 255, 0) if tag_id in args.permanent_ids else (255, 180, 0), 2)
                cv2.putText(view, str(tag_id), tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
            label = None
            if calibration:
                try:
                    label = make_label(calibration, detections, frame.intrinsic,
                                       view.shape, frame.timestamp_ns or time.time_ns())
                    draw_box(view, label)
                    if not status.startswith("Saved"):
                        status = (f"Mapped {len(calibration['T_table_tags'])}/{len(args.permanent_ids)}; "
                                  f"pose from {len(label['permanent_tags_used'])} tags, "
                                  f"error {label['tag_reprojection_rms_px']:.2f} px")
                except (ValueError, cv2.error) as exc:
                    status = str(exc)
            height, width = view.shape[:2]
            display = np.zeros((height + 110, width, 3), dtype=np.uint8)
            display[:height] = view
            status_text = status[:max(10, width // 10)]
            cv2.putText(display, status_text, (8, height + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
            box_button = "Reconfig BB [R]" if calibration else "Set BB [S]"
            if calibration:
                mapped_visible = sum(int(tag_id) in detections
                                     for tag_id in calibration["T_table_tags"])
                box_ready = (all(tag_id in detections for tag_id in args.setup_ids)
                             and mapped_visible >= 2)
            else:
                box_ready = (all(tag_id in detections for tag_id in args.setup_ids)
                             and sum(tag_id in detections for tag_id in args.permanent_ids) >= 2)
            reference_path = canonical_reference(
                args.output, args.centerpose_category, args.reference)
            captured_images = len(list((args.output / "images").glob("*.png")))
            scan_ready = calibration is not None and captured_images > 0
            button(display, 8, 175, box_button, box_ready, height + 22)
            button(display, 185, 350, "Capture [SPACE]",
                   label is not None, height + 22)
            new_ids = ([tag_id for tag_id in args.permanent_ids
                        if calibration and str(tag_id) not in calibration["T_table_tags"]
                        and tag_id in detections])
            button(display, 360, 505, "Add tags [A]", bool(calibration and new_ids), height + 22)
            button(display, 515, 615, "Quit [Q]", True, height + 22)
            scans_available = any((args.output / "reference_scans").glob("*.ply"))
            button(display, 8, 205, "Scan PLY [P]", scan_ready, height + 66)
            button(display, 215, 410, "Inspect scan [E]", scans_available, height + 66)
            reference_text = (f"Reference: {reference_path.parent.name}"
                              if reference_path else
                              f"Reference built from {captured_images} RGB-D captures after masks")
            cv2.putText(display, reference_text, (420, height + 91),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (90, 230, 110) if reference_path else (80, 170, 255),
                        1, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKeyEx(1) & 0xFF
            mouse_actions, actions[:] = actions[:], []
            box_click = any(height + 22 <= y < height + 60 and 8 <= x <= 175
                            for x, y in mouse_actions)
            set_box = calibration is None and (key in (ord("s"), ord("S")) or box_click)
            reconfigure = calibration is not None and (
                key in (ord("r"), ord("R")) or box_click)
            capture = key == 32 or any(
                height + 22 <= y < height + 60 and 185 <= x <= 350
                for x, y in mouse_actions)
            add_tags = key in (ord("a"), ord("A")) or any(
                height + 22 <= y < height + 60 and 360 <= x <= 505
                for x, y in mouse_actions)
            quit_now = key in (ord("q"), ord("Q"), 27) or any(
                height + 22 <= y < height + 60 and 515 <= x <= 615
                for x, y in mouse_actions)
            scan_reference = key in (ord("p"), ord("P")) or any(
                height + 66 <= y < height + 104 and 8 <= x <= 205
                for x, y in mouse_actions)
            edit_reference = key in (ord("e"), ord("E")) or any(
                height + 66 <= y < height + 104 and 215 <= x <= 410
                for x, y in mouse_actions)
            if quit_now:
                break
            if scan_reference:
                if not scan_ready:
                    status = "Cannot scan yet: set BB and capture at least one image first"
                    continue
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                destination = (args.output / "reference_scans" /
                               f"{args.centerpose_category}_{stamp}.ply").resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                expected_serial = camera.info["serial"]
                camera.close()
                cv2.destroyWindow(window)
                time.sleep(0.5)
                print("Starting TSDF reference scan. Keep the object stationary and move "
                      "the camera slowly around all visible sides.", flush=True)
                completed = subprocess.run(
                    reference_scan_command(args, destination, session_path), check=False)
                time.sleep(0.5)
                camera = open_camera(args.camera_backend, args.serial)
                if camera.info["serial"] != expected_serial:
                    camera.close()
                    raise RuntimeError("Reference scan reconnected to a different camera")
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window, click)
                if completed.returncode == 0 and destination.is_file():
                    status = ("Diagnostic TSDF PLY saved. Build the production reference "
                              "from masked calibrated RGB-D after capture.")
                else:
                    status = f"Reference scan failed (exit {completed.returncode}); see terminal log"
                print(status, flush=True)
                continue
            if edit_reference:
                if not scans_available:
                    status = "No raw reference scan yet; press P first"
                    continue
                if editor_process is None or editor_process.poll() is not None:
                    editor_process, editor_url = start_reference_editor(
                        args.output, args.centerpose_category,
                        args.reference_editor_port,
                        not args.no_open_reference_editor)
                elif not args.no_open_reference_editor and editor_url:
                    webbrowser.open(editor_url)
                status = "Diagnostic scan editor opened; production reference uses captured RGB-D"
            if set_box and calibration is None:
                try:
                    calibration = calibrate(detections, frame.intrinsic, args.setup_ids,
                                            args.permanent_ids,
                                            args.setup_tag_size_mm / 1000,
                                            args.permanent_tag_size_mm / 1000,
                                            tag_family=args.tag_family)
                    calibration["camera_serial"] = camera.info["serial"]
                    calibration["camera_backend"] = camera.info["backend"]
                    calibration["camera_name"] = camera.info["name"]
                    calibration["created_utc"] = datetime.now(timezone.utc).isoformat()
                    save_json(session_path, calibration)
                    roles = calibration["resolved_setup_roles"]
                    status = (f"Box set: base {roles['base_origin']}/{roles['base_x']}/"
                              f"{roles['base_y']}, top {roles['top']}. Remove all four setup tags.")
                    print(status, flush=True)
                except (ValueError, cv2.error) as exc:
                    status = f"Cannot set box: {exc}"
            if reconfigure and calibration:
                try:
                    updated = reconfigure_box(calibration, detections, frame.intrinsic)
                    updated["reconfigured_utc"] = datetime.now(timezone.utc).isoformat()
                    count, archive = apply_reconfiguration(args.output, calibration, updated)
                    calibration = updated
                    export_centerpose_dataset(
                        args.output, args.centerpose_category, args.centerpose_split)
                    roles = calibration["resolved_setup_roles"]
                    status = (f"BB revision {calibration['revision']}: base "
                              f"{roles['base_origin']}/{roles['base_x']}/{roles['base_y']}, "
                              f"top {roles['top']}; updated {count} labels")
                    print(f"{status}. Previous files: {archive}", flush=True)
                except (ValueError, cv2.error, OSError, KeyError) as exc:
                    status = f"Cannot reconfigure box: {exc}"
            if add_tags and calibration:
                try:
                    updated, added = register_visible_tags(calibration, detections, frame.intrinsic)
                    if added:
                        save_json(session_path, updated)
                        calibration = updated
                        status = f"Mapped new permanent tags {added}"
                    else:
                        status = "No new tag mapped; keep two mapped tags and new tags visible"
                except (ValueError, cv2.error) as exc:
                    status = f"Cannot add tags: {exc}"
            if capture:
                if label is None:
                    status = "Cannot capture: permanent tags or box pose unavailable"
                    continue
                if any(tag_id in detections for tag_id in args.setup_ids):
                    status = "Remove all four setup tags before capturing"
                    continue
                name = str(frame.timestamp_ns or time.time_ns())
                image_path = args.output / "images" / f"{name}.png"
                raw_path = args.output / "raw_images" / f"{name}.png"
                depth_path = args.output / "depths" / f"{name}.png"
                label_path = args.output / "labels" / f"{name}.json"
                if (image_path.exists() or raw_path.exists() or depth_path.exists()
                        or label_path.exists()):
                    status = "Timestamp already saved; try again"
                    continue
                raw_bgr = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
                model_rgb = remove_tags(frame.color, detections, args.permanent_ids)
                model_bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(raw_path), raw_bgr):
                    status = "Failed to write raw image"
                    continue
                if not cv2.imwrite(str(image_path), model_bgr):
                    status = "Failed to write image"
                    continue
                if not cv2.imwrite(str(depth_path), frame.depth):
                    status = "Failed to write aligned depth"
                    continue
                label["image"] = str(image_path.relative_to(args.output))
                label["raw_image"] = str(raw_path.relative_to(args.output))
                label["depth"] = str(depth_path.relative_to(args.output))
                label["depth_unit_m"] = 0.001
                label["depth_invalid_value"] = 0
                label["depth_aligned_to"] = "rectified color image"
                label["depth_range_m"] = [args.depth_min, args.depth_max]
                label["permanent_tags_inpainted"] = True
                save_json(label_path, label)
                try:
                    export_centerpose_sample(
                        args.output, label, args.centerpose_category, args.centerpose_split)
                except (ValueError, OSError, KeyError) as exc:
                    status = f"Saved audit label but CenterPose export failed: {exc}"
                    print(status, flush=True)
                    continue
                with (args.output / "samples.jsonl").open("a") as manifest:
                    manifest.write(json.dumps({"image": label["image"],
                                               "depth": label["depth"],
                                               "label": str(label_path.relative_to(args.output))}) + "\n")
                status = f"Saved {name}"
                print(status, flush=True)
    finally:
        camera.close()
        cv2.destroyAllWindows()
        if editor_process is not None and editor_process.poll() is None:
            editor_process.terminate()
            try:
                editor_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                editor_process.kill()


if __name__ == "__main__":
    main()
