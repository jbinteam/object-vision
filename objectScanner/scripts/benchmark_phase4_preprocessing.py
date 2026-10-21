#!/usr/bin/env python3
"""Compare legacy and current cloud preprocessing on read-only saved RGB-D views."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_ws/src/object_pose_tracking'))
from object_pose_tracking.geometry import CloudConfig, backproject, preprocess


def legacy_preprocess(points, config, colors):
    """Original axis-wise unique + add.at; retained only as a benchmark baseline."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if config.voxel_size > 0 and len(points):
        keys = np.floor(points / config.voxel_size).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(sums, inverse, points)
        points = (sums / counts[:, None]).astype(np.float32)
        color_sums = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(color_sums, inverse, colors)
        colors = np.rint(color_sums / counts[:, None]).clip(0, 255).astype(np.uint8)
    if config.outlier_neighbors > 0 and len(points) > 2:
        k = min(config.outlier_neighbors + 1, len(points))
        distances, _ = cKDTree(points).query(points, k=k, workers=1)
        means = distances[:, 1:].mean(axis=1)
        keep = means <= means.mean() + config.outlier_std_ratio * means.std()
        points, colors = points[keep], colors[keep]
    return np.ascontiguousarray(points), np.ascontiguousarray(colors)


def stats(values):
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/my_object_20260925_r2')
    parser.add_argument('--cases', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.cases < 1 or args.repeats < 1:
        parser.error('cases and repeats must be positive')
    root = args.dataset.resolve()
    paths = sorted((root / 'labels').glob('*.json'))
    if not paths:
        parser.error('dataset has no saved label JSON files')
    paths = [paths[index] for index in np.linspace(0, len(paths) - 1, min(args.cases, len(paths))).astype(int)]
    config = CloudConfig()  # Matches current Phase 4 object_cloud preprocessing.
    all_timings = [[], []]
    samples = []
    for path in paths:
        label = json.loads(path.read_text())
        if label.get('depth_unit_m') != .001:
            raise ValueError(f'{path}: benchmark expects saved uint16 millimetre depth')
        depth = cv2.imread(str(root / label['depth']), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(root / 'centerpose_masks/masks' / f'{path.stem}.png'), cv2.IMREAD_GRAYSCALE)
        bgr = cv2.imread(str(root / label['image']))
        if depth is None or mask is None or bgr is None:
            raise ValueError(f'{path}: missing saved depth, mask or image')
        points, colors = backproject(depth, mask, label['intrinsic'], config,
                                    rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        original = legacy_preprocess(points, config, colors)
        current = preprocess(points, config, colors)
        np.testing.assert_array_equal(original[0], current[0])
        np.testing.assert_array_equal(original[1], current[1])
        for _ in range(3):
            legacy_preprocess(points, config, colors)
            preprocess(points, config, colors)
        timings = [[], []]
        for repeat in range(args.repeats):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                started = time.perf_counter_ns()
                (legacy_preprocess, preprocess)[index](points, config, colors)
                timings[index].append((time.perf_counter_ns() - started) / 1e6)
        for target, source in zip(all_timings, timings):
            target.extend(source)
        samples.append({'sample': path.stem, 'image_shape': list(depth.shape),
                        'raw_points': len(points), 'filtered_points': len(current[0]),
                        'legacy_ms': stats(timings[0]), 'optimized_ms': stats(timings[1]),
                        'exact_points_colors_and_order': True})
    report = {'scope': 'Saved L515 RGB-D/masks; cloud preprocessing only, not Gemini/live end-to-end throughput',
              'dataset': str(root), 'configuration': asdict(config),
              'numpy': np.__version__, 'threads': {name: os.environ.get(name) for name in
                                                  ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
              'repeats_per_case': args.repeats, 'cases': samples,
              'legacy_ms': stats(all_timings[0]), 'optimized_ms': stats(all_timings[1])}
    report['median_speedup'] = report['legacy_ms']['median'] / report['optimized_ms']['median']
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
