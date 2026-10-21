#!/usr/bin/env python3
"""Tune PnP-initialized coarse-to-fine GICP without the ROS live pipeline."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_ws/src/object_pose_tracking'))
from object_pose_tracking.gicp import CoarseToFineGICP, GICPConfig  # noqa: E402


def load_transform(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get('T_camera_object', data.get('transform'))
    transform = np.asarray(data, dtype=float)
    if transform.size == 16:
        transform = transform.reshape(4, 4)
    return transform


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True,
                        help='JSON 4x4 T_camera_object or object containing that key')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-visibility-filter', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args(argv)
    observed_cloud = o3d.io.read_point_cloud(str(args.observed))
    observed = np.asarray(observed_cloud.points)
    config = GICPConfig(visibility_filter_enabled=not args.no_visibility_filter,
                        visibility_normal_facing=False)
    registration = CoarseToFineGICP(args.reference, config)
    initial = load_transform(args.initial)
    result = registration.register(observed, initial)
    report = {
        'accepted': result.accepted, 'reason': result.reason,
        'T_camera_object_refined': result.transform.tolist(),
        'fitness': result.fitness, 'rmse_m': result.rmse,
        'overlap': result.overlap, 'correspondences': result.correspondences,
        'visible_reference_points': result.visible_reference_points,
        'level_ms': result.level_ms, 'runtime_ms': result.total_ms,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n')
    if args.visualize:
        before = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            registration.reference_points @ initial[:3, :3].T + initial[:3, 3]))
        after = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            registration.reference_points @ result.transform[:3, :3].T + result.transform[:3, 3]))
        before.paint_uniform_color([1., .2, .8])
        after.paint_uniform_color([1., .8, .1])
        observed_cloud.paint_uniform_color([.1, .8, 1.])
        o3d.visualization.draw_geometries([observed_cloud, before, after])
    return 0 if result.accepted else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f'Offline GICP failed: {error}')
