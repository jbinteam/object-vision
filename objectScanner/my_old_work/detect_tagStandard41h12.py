#!/usr/bin/env python3
"""Detect tagStandard41h12 AprilTags in an image, video, or webcam.

This is a focused wrapper around the tagStandard41h12 family (the newer
"standard" AprilTag with a recessed white border and 2115 unique ids).
Optionally estimates 6-DoF pose when camera intrinsics and the physical
tag size are supplied.

Examples:
    python detect_tagStandard41h12.py --image apriltag/samples/tagStandard41h12/tagStandard41h12_00003.png
    python detect_tagStandard41h12.py --camera 0
    python detect_tagStandard41h12.py --camera 0 --calib apriltag/my_intrinsics.json --tag-size 0.16

Requires:
    pip install pupil-apriltags opencv-python numpy
"""

import argparse
import sys

import cv2
import numpy as np

import intrinsics

try:
    from pupil_apriltags import Detector
except ImportError:
    sys.exit(
        "Missing dependency. Install it with:\n"
        "    pip install pupil-apriltags opencv-python numpy"
    )

FAMILY = "tagStandard41h12"


def make_detector(nthreads: int) -> Detector:
    """Build a Detector locked to the tagStandard41h12 family."""
    return Detector(
        families=FAMILY,
        nthreads=nthreads,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )


def euler_deg(R):
    """Rotation matrix -> roll/pitch/yaw degrees (XYZ)."""
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def draw_axes(frame, K, dist, R, t, length) -> None:
    """Project tag coordinate axes (x=red, y=green, z=blue)."""
    rvec, _ = cv2.Rodrigues(R)
    axis = np.float32([[0, 0, 0], [length, 0, 0],
                       [0, length, 0], [0, 0, length]])
    pts, _ = cv2.projectPoints(axis, rvec, t.reshape(3, 1), K, dist)
    pts = pts.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    cv2.line(frame, origin, tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(frame, origin, tuple(pts[2]), (0, 255, 0), 2)
    cv2.line(frame, origin, tuple(pts[3]), (255, 0, 0), 2)


def draw(frame, detections, K=None, dist=None, tag_size=None) -> None:
    """Annotate detections with outline, id, margin, and pose axes."""
    for det in detections:
        corners = det.corners.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]),
                     (0, 255, 0), 2)
        cx, cy = int(det.center[0]), int(det.center[1])
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"id={det.tag_id} m={det.decision_margin:.0f}",
                    (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if det.pose_t is not None and K is not None:
            draw_axes(frame, K, dist, det.pose_R, det.pose_t, tag_size * 0.5)
            cv2.putText(frame, f"{np.linalg.norm(det.pose_t):.2f} m",
                        (corners[0][0], corners[0][1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def detect(detector, frame, camera_params=None, tag_size=None):
    """Detect tagStandard41h12 tags; estimate pose if intrinsics given."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if camera_params is not None and tag_size is not None:
        return detector.detect(gray, estimate_tag_pose=True,
                               camera_params=camera_params, tag_size=tag_size)
    return detector.detect(gray)


def print_detection(det) -> None:
    line = (f"  id={det.tag_id} center=({det.center[0]:.1f}, {det.center[1]:.1f}) "
            f"margin={det.decision_margin:.1f}")
    if det.pose_t is not None:
        t = det.pose_t.ravel()
        r, p, y = euler_deg(det.pose_R)
        line += (f"\n      pos(m): x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f}"
                 f"  dist={np.linalg.norm(t):.3f}"
                 f"\n      rot(deg): roll={r:.1f} pitch={p:.1f} yaw={y:.1f}")
    print(line)


def run_image(detector, path, save, show, camera_params, K, dist, tag_size) -> int:
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Could not read image: {path}")
    detections = detect(detector, frame, camera_params, tag_size)
    print(f"Detected {len(detections)} {FAMILY} tag(s).")
    for det in detections:
        print_detection(det)
    draw(frame, detections, K, dist, tag_size)
    if save:
        cv2.imwrite(save, frame)
        print(f"Saved annotated image to {save}")
    if show:
        cv2.imshow(FAMILY, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return len(detections)


def run_stream(detector, source, save, camera_params, K, dist, tag_size) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")
    writer = None
    print("Press 'q' in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detect(detector, frame, camera_params, tag_size)
        draw(frame, detections, K, dist, tag_size)
        cv2.putText(frame, f"{FAMILY} tags: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if save and writer is None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer is not None:
            writer.write(frame)
        cv2.imshow(FAMILY, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {save}")
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Detect {FAMILY} AprilTags.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to an image file.")
    src.add_argument("--video", help="Path to a video file.")
    src.add_argument("--camera", type=int, help="Camera index (e.g. 0).")
    parser.add_argument("--nthreads", type=int, default=4, help="Detector threads.")
    parser.add_argument("--save", help="Path to save annotated output.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open a display window (headless).")
    parser.add_argument("--camera-params", type=float, nargs=4,
                        metavar=("FX", "FY", "CX", "CY"),
                        help="Camera intrinsics for pose estimation.")
    parser.add_argument("--calib",
                        help="Intrinsics file (.json/.yml/.xml/.npz). Overrides the "
                             f"shared default ({intrinsics.DEFAULT_CALIB}).")
    parser.add_argument("--no-default-calib", action="store_true",
                        help="Do not fall back to the shared default intrinsics file.")
    parser.add_argument("--tag-size", type=float,
                        help="Physical tag side length in metres (required for pose).")
    args = parser.parse_args()

    camera_params, K, dist = intrinsics.resolve(
        calib=args.calib,
        camera_params=args.camera_params,
        use_default=not args.no_default_calib,
    )
    if camera_params is not None and args.tag_size is None:
        sys.exit("Pose estimation needs --tag-size (tag side length in metres).")
    if camera_params is None:
        print("Pose estimation disabled (provide --camera-params or --calib, plus --tag-size).")

    detector = make_detector(args.nthreads)

    if args.image:
        run_image(detector, args.image, args.save, not args.no_show,
                  camera_params, K, dist, args.tag_size)
    elif args.video:
        run_stream(detector, args.video, args.save,
                   camera_params, K, dist, args.tag_size)
    else:
        run_stream(detector, args.camera, args.save,
                   camera_params, K, dist, args.tag_size)


if __name__ == "__main__":
    main()
