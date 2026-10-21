#!/usr/bin/env python3
"""Run robust PnP independently from ROS and optionally draw reprojections."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_ws/src/object_pose_tracking'))
from object_pose_tracking.pnp import PnPConfig, RobustPnP, load_object_keypoints  # noqa: E402


def load_observation(path, count):
    data = json.loads(Path(path).read_text())
    rows = data.get('keypoints', data.get('points'))
    if not isinstance(rows, list):
        raise ValueError('Observation JSON needs a keypoints list')
    if rows and isinstance(rows[0], dict):
        output = np.full((count, 4), [np.nan, np.nan, 0., 0.])
        for row in rows:
            index = int(row['id'])
            output[index] = [row['u'], row['v'], row.get('confidence', 1.),
                             row.get('valid', True)]
    else:
        output = np.asarray(rows, dtype=float)
        if output.shape == (count, 2):
            output = np.column_stack((output, np.ones((count, 2))))
        elif output.shape == (count, 3):
            output = np.column_stack((output, np.ones(count)))
    return data, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-keypoints', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True,
                        help='JSON with keypoints and camera_matrix/K')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--keypoint-confidence', type=float, default=.35)
    parser.add_argument('--max-reprojection-error', type=float, default=5.)
    args = parser.parse_args(argv)
    points, _ = load_object_keypoints(args.object_keypoints)
    data, detected = load_observation(args.observation, len(points))
    intrinsic = data.get('camera_matrix', data.get('K'))
    if intrinsic is None:
        raise ValueError('Observation JSON needs camera_matrix or K')
    config = PnPConfig(keypoint_confidence_threshold=args.keypoint_confidence,
                       max_mean_reprojection_error=args.max_reprojection_error,
                       depth_validation_enabled=False)
    result = RobustPnP(points, config).estimate(
        detected, intrinsic, data.get('distortion_coefficients', data.get('distortion')))
    report = {
        'accepted': result.accepted, 'reason': result.reason,
        'T_camera_object': result.transform.tolist(),
        'valid_keypoints': result.valid_keypoints, 'pnp_inliers': result.inlier_count,
        'inlier_ids': result.inlier_ids.tolist(),
        'mean_reprojection_error_px': result.mean_reprojection_error,
        'median_reprojection_error_px': result.median_reprojection_error,
        'max_reprojection_error_px': result.max_reprojection_error,
        'pnp_ms': result.pnp_ms,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n')
    if args.image and args.overlay:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f'Cannot read image: {args.image}')
        for index, row in enumerate(detected):
            if row[3] and np.isfinite(row[:2]).all():
                point = tuple(np.rint(row[:2]).astype(int))
                cv2.circle(image, point, 5, (0, 255, 0), 2)
                cv2.putText(image, str(index), (point[0] + 4, point[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
        for point in result.projected_keypoints:
            if np.isfinite(point).all():
                cv2.drawMarker(image, tuple(np.rint(point).astype(int)), (255, 0, 255),
                               cv2.MARKER_CROSS, 9, 2)
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.overlay), image):
            raise OSError(f'Cannot write overlay: {args.overlay}')
    return 0 if result.accepted else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f'Offline PnP failed: {error}')
