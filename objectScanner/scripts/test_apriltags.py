#!/usr/bin/env python3
"""Minimal AprilTag test for a saved image or a live RGB-D camera."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import open_camera  # noqa: E402
from turntable_dataset import (TAG_FAMILIES, detect_tags, make_detector,  # noqa: E402
                               pose_from_correspondences, project,
                               tag_corners, transform_points)


CORNER_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255))


def outlined_text(view, text, origin, scale=.55, color=(255, 255, 255)):
    cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def ids_arg(value):
    try:
        ids = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer tag IDs") from exc
    if not ids or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("Tag IDs must be unique")
    return ids


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Detect tags in one image")
    source.add_argument("--camera", action="store_true", help="Use the selected live RGB-D camera")
    source.add_argument("--l515", action="store_true", help="Use the legacy L515 (sets --camera-backend realsense)")
    parser.add_argument("--camera-backend", choices=("orbbec", "realsense", "auto"), default="orbbec")
    parser.add_argument("--family", choices=TAG_FAMILIES, default="tagStandard41h12",
                        help="Printed tag family (default: tagStandard41h12)")
    parser.add_argument("--tag-size-mm", type=float,
                        help="One size for every ID; overrides the two group sizes")
    parser.add_argument("--setup-tag-size-mm", type=float, default=14,
                        help="Size for setup IDs (default: 14)")
    parser.add_argument("--permanent-tag-size-mm", type=float, default=28,
                        help="Size for permanent IDs (default: 28)")
    parser.add_argument("--setup-ids", type=ids_arg, default=(1, 2, 3, 4))
    parser.add_argument("--permanent-ids", type=ids_arg, default=(5, 6, 7, 8, 9, 10))
    parser.add_argument("--serial", help="Camera serial when more than one is connected")
    parser.add_argument("--output", type=Path,
                        help="Image output path, or live snapshot directory")
    parser.add_argument("--no-display", action="store_true",
                        help="Image mode only: print results without opening a window")
    args = parser.parse_args()
    if args.tag_size_mm is not None and not 5 <= args.tag_size_mm <= 300:
        parser.error("--tag-size-mm must be between 5 and 300")
    if args.image and args.tag_size_mm is not None:
        parser.error("Tag size is used for live camera pose; image mode has no camera intrinsics")
    for value in (args.setup_tag_size_mm, args.permanent_tag_size_mm):
        if not 5 <= value <= 300:
            parser.error("Group tag sizes must be between 5 and 300 mm")
    if set(args.setup_ids) & set(args.permanent_ids):
        parser.error("Setup and permanent IDs must be distinct")
    if (args.camera or args.l515) and args.no_display:
        parser.error("--no-display is only available with --image")
    return args


def estimate_poses(detections, intrinsic, tag_sizes_m):
    poses = {}
    if intrinsic is None:
        return poses
    for tag_id, pixels in detections.items():
        tag_size_m = tag_sizes_m.get(tag_id)
        if tag_size_m is None:
            continue
        try:
            pose, error = pose_from_correspondences(
                tag_corners(tag_size_m), pixels, intrinsic, square=True)
            poses[tag_id] = pose, error, tag_size_m
        except (ValueError, cv2.error):
            pass
    return poses


def annotate(rgb, detections, intrinsic=None, tag_sizes_m=None):
    view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    poses = estimate_poses(detections, intrinsic, tag_sizes_m or {})
    for tag_id in sorted(detections):
        corners = np.rint(detections[tag_id]).astype(int)
        cv2.polylines(view, [corners.astype(np.int32)], True, (255, 255, 255), 2)
        for index, corner in enumerate(corners):
            point = tuple(corner)
            cv2.circle(view, point, 5, CORNER_COLORS[index], -1)
            outlined_text(view, str(index), (point[0] + 5, point[1] - 5),
                          .5, CORNER_COLORS[index])
        center = tuple(np.rint(detections[tag_id].mean(axis=0)).astype(int))
        text = f"ID {tag_id}"
        if tag_id in poses:
            pose, error, tag_size_m = poses[tag_id]
            text += f"  z={pose[2, 3]:.3f}m  err={error:.2f}px"
            length = tag_size_m * .5
            axes = np.array([[0, 0, 0], [length, 0, 0],
                             [0, length, 0], [0, 0, length]], dtype=np.float64)
            try:
                uv = np.rint(project(intrinsic, transform_points(pose, axes))).astype(int)
                for end, color in zip(uv[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    cv2.line(view, tuple(uv[0]), tuple(end), color, 2)
            except ValueError:
                pass
        outlined_text(view, text, (center[0] + 8, center[1]), .55)
    outlined_text(view, f"{len(detections)} tag(s)", (10, 28), .75, (0, 255, 255))
    return view, poses


def print_results(detections, poses):
    if not detections:
        print("No tags detected.")
        return
    print(f"Detected {len(detections)} tag(s): {sorted(detections)}")
    for tag_id in sorted(detections):
        center = detections[tag_id].mean(axis=0)
        line = f"  ID {tag_id}: center=({center[0]:.1f}, {center[1]:.1f})"
        if tag_id in poses:
            pose, error, tag_size_m = poses[tag_id]
            x, y, z = pose[:3, 3]
            line += (f"  size={tag_size_m * 1000:g}mm"
                     f"  xyz=({x:.4f}, {y:.4f}, {z:.4f}) m  error={error:.2f} px")
        print(line)


def test_image(args, detector):
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    detections = detect_tags(rgb, detector)
    view, poses = annotate(rgb, detections)
    print_results(detections, poses)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), view):
            raise SystemExit(f"Cannot write output: {args.output}")
        print(f"Saved annotated image: {args.output}")
    if not args.no_display:
        cv2.imshow("AprilTag image test | Q closes", view)
        while cv2.waitKey(0) & 0xFF not in (ord("q"), ord("Q"), 27):
            pass
        cv2.destroyAllWindows()
    return 0 if detections else 2


def test_camera(args, detector):
    camera = open_camera("realsense" if args.l515 else args.camera_backend, args.serial)
    snapshot_dir = args.output or Path("output/apriltag_test")
    previous_ids = None
    if args.tag_size_mm:
        tag_sizes_m = {tag_id: args.tag_size_mm / 1000
                       for tag_id in args.setup_ids + args.permanent_ids}
    else:
        tag_sizes_m = {tag_id: args.setup_tag_size_mm / 1000 for tag_id in args.setup_ids}
        tag_sizes_m.update({tag_id: args.permanent_tag_size_mm / 1000
                            for tag_id in args.permanent_ids})
    print("Live detector started. Press S to save raw and annotated images; Q/ESC quits.")
    if args.tag_size_mm:
        print(f"Pose size override: every configured ID = {args.tag_size_mm:g} mm.")
    else:
        print(f"Pose sizes: setup IDs {args.setup_ids} = {args.setup_tag_size_mm:g} mm; "
              f"permanent IDs {args.permanent_ids} = {args.permanent_tag_size_mm:g} mm.")
    try:
        while True:
            frame = camera.read(.15, 3.0, 1.0, 1)
            if frame is None:
                continue
            detections = detect_tags(frame.color, detector)
            view, poses = annotate(frame.color, detections, frame.intrinsic, tag_sizes_m)
            ids = tuple(sorted(detections))
            if ids != previous_ids:
                print_results(detections, poses)
                previous_ids = ids
            cv2.imshow("AprilTag RGB-D test | S saves | Q quits", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                name = str(frame.timestamp_ns or time.time_ns())
                raw_path = snapshot_dir / f"{name}_raw.png"
                marked_path = snapshot_dir / f"{name}_detected.png"
                cv2.imwrite(str(raw_path), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(marked_path), view)
                print(f"Saved {raw_path} and {marked_path}")
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return 0


def main():
    args = arguments()
    detector = make_detector(args.family)
    return test_camera(args, detector) if (args.camera or args.l515) else test_image(args, detector)


if __name__ == "__main__":
    raise SystemExit(main())
