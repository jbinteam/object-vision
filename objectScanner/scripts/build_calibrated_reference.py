#!/usr/bin/env python3
"""Fuse calibrated masked RGB-D captures directly in the PnP box_center frame.

Each saved T_camera_object is the Open3D TSDF world-to-camera extrinsic because
the TSDF world is the calibrated object frame. No moving-camera SLAM frame or
manual reference-axis assignment enters this reference.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nocs_dataset import extract_masked_object_points  # noqa: E402
from scripts.generate_centerpose_reference import (  # noqa: E402
    object_keypoints_yaml, preflight, read_inputs, resolve_augmentation,
)


def rigid(value):
    transform = np.asarray(value, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Capture T_camera_object must be a finite rigid transform')
    return transform


def build(dataset, masks, output, voxel_mm=2.5, truncation_mm=12.5,
          box_margin_mm=3.0, min_views=8, min_points=1000, excluded_stems=()):
    dataset, masks, manifest_path, manifest, records = resolve_augmentation(dataset, masks)
    calibration = json.loads((dataset / 'calibration.json').read_text())
    items, dimensions, readiness = preflight(dataset, masks, records)
    if not readiness['ready']:
        raise ValueError(f'Incomplete capture/mask pairs: {readiness["missing_counts"]}')
    if (calibration.get('object_frame') != 'box_center'
            or dimensions is None
            or not np.allclose(dimensions, calibration['box_dimensions_m'], atol=1e-8, rtol=0)):
        raise ValueError('Capture box differs from the current calibration')
    if not (0 < voxel_mm <= 10 and voxel_mm < truncation_mm <= 50
            and 0 <= box_margin_mm <= 20 and min_views >= 1 and min_points >= 30):
        raise ValueError('Invalid TSDF or reference quality settings')
    excluded = set(excluded_stems)
    items = [item for item in items if item[0] not in excluded]
    if len(items) < min_views:
        raise ValueError(f'Need at least {min_views} calibrated views, found {len(items)}')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Reference already exists: {output}')

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_mm / 1000, sdf_trunc=truncation_mm / 1000,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    rows = []
    for stem, _label_path, label, *_ in items:
        if (label.get('object_frame') != 'box_center'
                or label.get('calibration_revision') != calibration.get('revision')
                or label.get('camera_frame') not in ('color_optical_frame',
                                                     'camera_color_optical_frame')
                or label.get('depth_aligned_to') != 'rectified color image'):
            raise ValueError(f'Capture frame differs from calibration: {stem}')
        camera_object = rigid(label['T_camera_object'])
        image_bgr, depth, mask = read_inputs((stem, _label_path, label, *_))
        _foreground, ys, xs, _camera, _object, metrics = extract_masked_object_points(
            depth, mask, label, box_tolerance_ratio=0.08)
        if len(xs) < 100:
            raise ValueError(f'Too few valid masked depth pixels in {stem}: {len(xs)}')
        filtered_depth = np.zeros_like(depth)
        filtered_depth[ys, xs] = depth[ys, xs]
        intrinsic = np.asarray(label['intrinsic'], dtype=np.float64)
        height, width = depth.shape
        camera = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic[0, 0], intrinsic[1, 1],
            intrinsic[0, 2], intrinsic[1, 2])
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(filtered_depth),
            depth_scale=1.0 / float(label['depth_unit_m']),
            depth_trunc=float(label['depth_range_m'][1]),
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, camera, camera_object)
        rows.append(dict(stem=stem, valid_depth_pixels=int(len(xs)),
                         tag_reprojection_rms_px=label.get('tag_reprojection_rms_px'),
                         box_consistency=metrics['box_consistency']))

    cloud = volume.extract_point_cloud()
    points = np.asarray(cloud.points)
    if len(points) < min_points:
        raise ValueError(f'TSDF extracted only {len(points)} points')
    half = dimensions / 2 + box_margin_mm / 1000
    inside = np.all(np.abs(points) <= half, axis=1)
    cloud = cloud.select_by_index(np.flatnonzero(inside))
    points_after_crop = len(cloud.points)
    if points_after_crop < min_points:
        raise ValueError(f'Calibrated box contains only {points_after_crop} TSDF points')
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(cloud.points) < min_points:
        raise ValueError(f'Only {len(cloud.points)} points remain after outlier filtering')
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.01, max_nn=30))
    normals = np.asarray(cloud.normals)
    points = np.asarray(cloud.points)
    normals[np.einsum('ij,ij->i', normals, points) < 0] *= -1

    metadata = dict(
        schema_version=4, name=output.name, frame_id='box_center', object_frame='box_center',
        units='m', method='calibrated_masked_rgbd_tsdf',
        transform_convention='P_camera = T_camera_object * P_object',
        reference_already_in_object_frame=True,
        T_scan_obj=np.eye(4).tolist(), T_obj_scan=np.eye(4).tolist(),
        box_dimensions_m=dimensions.tolist(), calibration_revision=calibration['revision'],
        source=dict(dataset=str(dataset), augmentation_manifest=str(manifest_path),
                    category=manifest.get('category'), views=len(rows)),
        voxel_size_mm=voxel_mm, sdf_truncation_mm=truncation_mm,
        box_margin_mm=box_margin_mm, points_before_crop=int(len(inside)),
        points_after_crop=points_after_crop, points=len(cloud.points),
        views=rows, created_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-building-', dir=output.parent))
    try:
        if not o3d.io.write_point_cloud(str(stage / 'reference.ply'), cloud):
            raise OSError('Could not write calibrated reference PLY')
        (stage / 'frame.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (stage / 'object_keypoints.yaml').write_text(object_keypoints_yaml(dimensions))
        (stage / 'README.txt').write_text(
            'This reference was fused from masked RGB-D captures with their calibrated\n'
            'T_camera_object poses. It and the eight PnP corners use box_center.\n'
            'No moving-camera scan or manual reference-axis assignment was used.\n')
        with zipfile.ZipFile(stage / 'reference_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
            for name in ('reference.ply', 'frame.json', 'object_keypoints.yaml', 'README.txt'):
                bundle.write(stage / name, arcname=f'{output.name}/{name}')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'Calibrated reference: {output / "reference.ply"}\n'
          f'Views: {len(rows)}; points: {len(cloud.points)}; frame: box_center')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voxel-mm', type=float, default=2.5)
    parser.add_argument('--truncation-mm', type=float, default=12.5)
    parser.add_argument('--box-margin-mm', type=float, default=3.0)
    parser.add_argument('--min-views', type=int, default=8)
    parser.add_argument('--min-points', type=int, default=1000)
    parser.add_argument('--exclude-stems', help='Comma-separated capture stems to exclude for validation')
    args = parser.parse_args()
    try:
        build(args.dataset, args.masks, args.output, args.voxel_mm,
              args.truncation_mm, args.box_margin_mm, args.min_views, args.min_points,
              args.exclude_stems.split(',') if args.exclude_stems else ())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f'Calibrated reference build failed: {exc}\n')


if __name__ == '__main__':
    main()
