#!/usr/bin/env python3
"""Align a scanned reference to the capture dataset's canonical object origin.

The captured depth clouds are transformed into ``object_frame`` with their
AprilTag-derived ``T_camera_object`` labels, then fused only for registration.
By default the scan's calibrated axes stay locked: the estimated rotation is
reported but only the residual translation is applied to the reference PLY.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_ws/src/object_pose_tracking'))
from object_pose_tracking.geometry import CloudConfig, backproject  # noqa: E402


def captured_object_cloud(dataset, masks):
    config = CloudConfig(outlier_neighbors=0)
    calibration = json.loads((dataset / 'calibration.json').read_text())
    dimensions = np.asarray(calibration['box_dimensions_m'], dtype=np.float64)
    clouds = []
    used = []
    for label_path in sorted((dataset / 'labels').glob('*.json')):
        stem = label_path.stem
        depth_path = dataset / 'depths' / f'{stem}.png'
        mask_path = masks / f'{stem}.png'
        if not depth_path.is_file() or not mask_path.is_file():
            continue
        label = json.loads(label_path.read_text())
        label_dimensions = np.asarray(label.get('box_dimensions_m', []), dtype=np.float64)
        if (label.get('object_frame') != calibration.get('object_frame')
                or label.get('calibration_revision') != calibration.get('revision')
                or label.get('camera_frame') not in ('color_optical_frame',
                                                     'camera_color_optical_frame')
                or label.get('depth_aligned_to') != 'rectified color image'
                or label_dimensions.shape != (3,)
                or not np.allclose(label_dimensions, dimensions,
                                   atol=1e-8, rtol=0)):
            raise ValueError(f'Capture frame differs from current calibration: {label_path}')
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if depth is None or mask is None:
            continue
        camera = backproject(depth, mask, label['intrinsic'], config, '16UC1')
        transform = np.asarray(label['T_camera_object'], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f'Invalid T_camera_object in {label_path}')
        # Row-vector inverse of P_camera = R * P_object + t.
        clouds.append((camera - transform[:3, 3]) @ transform[:3, :3])
        used.append(stem)
    if not clouds:
        raise ValueError('No complete label/depth/mask capture triples found')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(clouds)))
    cloud = cloud.voxel_down_sample(.002)
    cloud, _ = cloud.remove_statistical_outlier(20, 2.0)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(.02, 30))
    return cloud, used


def estimate_alignment(reference, target):
    reference.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(.02, 30))
    transform = np.eye(4)
    levels = ((.010, .040, 60), (.005, .025, 50), (.002, .012, 40))
    metrics = []
    for voxel, distance, iterations in levels:
        source_level = reference.voxel_down_sample(voxel)
        target_level = target.voxel_down_sample(voxel)
        result = o3d.pipelines.registration.registration_generalized_icp(
            source_level, target_level, distance, transform,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations))
        transform = np.asarray(result.transformation).copy()
        metrics.append(dict(voxel_size_m=voxel,
                            max_correspondence_distance_m=distance,
                            max_iterations=iterations,
                            fitness=float(result.fitness),
                            rmse_m=float(result.inlier_rmse)))
    return transform, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--masks', type=Path,
                        help='Default: DATASET/centerpose_masks/masks')
    parser.add_argument('--output', type=Path,
                        help='Default: reference_aligned.ply beside --reference')
    parser.add_argument('--allow-rotation', action='store_true',
                        help='Also apply the estimated rotation; default locks the calibrated axes')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    dataset = args.dataset.resolve()
    reference_path = args.reference.resolve()
    masks = (args.masks or dataset / 'centerpose_masks/masks').resolve()
    output = (args.output or reference_path.with_name('reference_aligned.ply')).resolve()
    if not reference_path.is_file() or not masks.is_dir():
        raise ValueError('Reference PLY or mask directory is missing')
    if output.exists() and not args.force:
        raise FileExistsError(f'{output} exists; pass --force to replace it')

    reference = o3d.io.read_point_cloud(str(reference_path))
    if len(reference.points) < 30:
        raise ValueError('Reference cloud has too few points')
    target, samples = captured_object_cloud(dataset, masks)
    estimated, levels = estimate_alignment(reference, target)
    angle = float(np.degrees(np.arccos(np.clip(
        (np.trace(estimated[:3, :3]) - 1) / 2, -1, 1))))
    distance = float(np.linalg.norm(estimated[:3, 3]))
    if distance > .08 or angle > 25:
        raise ValueError(
            f'Implausible reference correction: {distance:.3f} m, {angle:.1f} deg')

    applied = estimated.copy()
    if not args.allow_rotation:
        applied[:3, :3] = np.eye(3)
    reference.transform(applied)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), reference):
        raise OSError(f'Could not write {output}')
    report = dict(
        schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_reference=str(reference_path),
        output_reference=str(output),
        dataset=str(dataset),
        masks=str(masks),
        samples=len(samples),
        target_points=len(target.points),
        reference_points=len(reference.points),
        object_frame='box_center',
        transform_convention='P_camera = T_camera_object * P_object',
        axis_policy=('estimated rigid correction' if args.allow_rotation
                     else 'calibrated axes locked; residual translation only'),
        estimated_T_object_corrected_from_object_scan=estimated.tolist(),
        applied_T_object_corrected_from_object_scan=applied.tolist(),
        estimated_translation_m=distance,
        estimated_rotation_deg=angle,
        levels=levels,
        sample_ids=samples)
    sidecar = output.with_suffix('.json')
    sidecar.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'output_reference', 'samples', 'target_points', 'reference_points',
        'axis_policy', 'estimated_translation_m', 'estimated_rotation_deg')}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f'Reference alignment failed: {error}')
