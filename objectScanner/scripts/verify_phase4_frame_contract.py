#!/usr/bin/env python3
"""Verify the PnP keypoints and GICP reference share one calibrated object frame."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rigid(value, name):
    transform = np.asarray(value, dtype=np.float64)
    require(transform.shape == (4, 4) and np.isfinite(transform).all(), f'{name} must be finite 4x4')
    require(np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8), f'{name} has invalid last row')
    rotation = transform[:3, :3]
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-6), f'{name} is not rigid')
    return transform


def verify(dataset, category, calibration_path=None, keypoints_path=None):
    dataset = Path(dataset).resolve()
    reference_dir = dataset / 'references' / category
    raw_reference = reference_dir / 'reference.ply'
    aligned_reference = reference_dir / 'reference_aligned.ply'
    keypoints_path = Path(keypoints_path) if keypoints_path else dataset / 'yolo_seg_pose_onthefly/object_keypoints.yaml'
    calibration_path = Path(calibration_path) if calibration_path else dataset / 'calibration.json'
    calibration = json.loads(calibration_path.read_text())
    frame = json.loads((reference_dir / 'frame.json').read_text())
    keypoints = yaml.safe_load(keypoints_path.read_text())

    require(calibration['object_frame'] == frame['frame_id'] == keypoints['object_frame']
            == 'box_center', 'Object frame IDs disagree')
    require(frame['units'] == keypoints['units'] == 'm', 'Reference/keypoint units disagree')
    dimensions = np.asarray(calibration['box_dimensions_m'], dtype=np.float64)
    require(dimensions.shape == (3,) and np.isfinite(dimensions).all()
            and np.all(dimensions > 0), 'Invalid calibration box dimensions')
    entries = keypoints['object_keypoints']
    require([entry['id'] for entry in entries] == list(range(8)), 'Keypoint order is not 0..7')
    corners = np.asarray([entry['xyz'] for entry in entries], dtype=np.float64)
    expected = np.asarray([[x * dimensions[0] / 2, y * dimensions[1] / 2,
                            z * dimensions[2] / 2]
                           for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)])
    require(np.allclose(corners, expected, atol=1e-8, rtol=0),
            'PnP keypoints differ from the calibrated box')

    if frame.get('method') == 'calibrated_masked_rgbd_tsdf':
        reference_keypoints = yaml.safe_load(
            (reference_dir / 'object_keypoints.yaml').read_text())
        reference_corners = np.asarray(
            [entry['xyz'] for entry in reference_keypoints['object_keypoints']],
            dtype=np.float64)
        require(reference_keypoints['object_frame'] == 'box_center'
                and reference_corners.shape == corners.shape
                and np.allclose(reference_corners, corners, atol=1e-8, rtol=0),
                'Reference and PnP keypoint coordinates differ')
        require(frame.get('calibration_revision') == calibration.get('revision'),
                'Reference was built from a different calibration revision')
        require(np.allclose(frame['box_dimensions_m'], dimensions, atol=1e-8, rtol=0),
                'Reference and PnP box dimensions differ')
        require(Path(frame['source']['dataset']).resolve() == dataset
                and Path(frame['source']['augmentation_manifest']).resolve()
                == (dataset / 'centerpose_masks/manifest.json').resolve(),
                'Reference provenance points to another capture or mask set')
        require(frame['source']['category'] == category, 'Reference category differs')
        require(np.allclose(rigid(frame['T_scan_obj'], 'reference transform'), np.eye(4), atol=1e-8)
                and np.allclose(rigid(frame['T_obj_scan'], 'inverse reference transform'),
                                np.eye(4), atol=1e-8),
                'Calibrated RGB-D reference must already be in box_center')
        points = np.asarray(o3d.io.read_point_cloud(str(raw_reference)).points)
        require(points.ndim == 2 and points.shape[1] == 3
                and len(points) == frame['points'] and len(points) >= 30,
                'Calibrated reference point count differs from metadata')
        half = dimensions / 2 + frame['box_margin_mm'] / 1000 + 1e-6
        require(np.isfinite(points).all() and np.all(np.abs(points) <= half),
                'Calibrated reference leaves the calibrated box')
        require(not aligned_reference.exists(),
                'Depth-fused reference must not have a second alignment transform')
        print(f'Frame contract OK: {len(corners)} PnP corners and {len(points)} calibrated '
              'RGB-D reference points in box_center.')
        return

    alignment = json.loads((reference_dir / 'reference_aligned.json').read_text())
    require(alignment['object_frame'] == 'box_center', 'Reference alignment frame differs')
    require(np.allclose(frame['source']['suggested_crop']['box_dimensions_m'], dimensions,
                        atol=1e-8, rtol=0), 'Scan crop used different box dimensions')

    scan_path = (dataset / 'reference_scans' / frame['source']['name']).resolve()
    require(scan_path.is_file(), f'Missing source scan: {scan_path}')
    require(hashlib.sha256(scan_path.read_bytes()).hexdigest() == frame['source']['sha256'],
            'Source scan differs from the PLY used to export the reference')
    scan_sidecar = json.loads(scan_path.with_name(scan_path.stem + '_scan.json').read_text())
    anchor = scan_sidecar['reference_anchor']
    require(anchor['frame_id'] == 'box_center', 'Scan anchor uses a different object frame')
    scan_object = rigid(anchor['T_scan_object'], 'scan anchor')
    export_scan_object = rigid(frame['T_scan_obj'], 'reference export transform')
    object_scan = rigid(frame['T_obj_scan'], 'inverse reference export transform')
    require(np.allclose(scan_object, export_scan_object, atol=1e-8, rtol=0)
            and np.allclose(object_scan @ scan_object, np.eye(4), atol=1e-8),
            'Reference export does not invert the calibrated scan anchor')

    require(Path(alignment['source_reference']).resolve() == raw_reference.resolve()
            and Path(alignment['output_reference']).resolve() == aligned_reference.resolve(),
            'Alignment report names different reference files')
    correction = rigid(alignment['applied_T_object_corrected_from_object_scan'],
                       'reference correction')
    require(np.allclose(correction[:3, :3], np.eye(3), atol=1e-8),
            'Reference correction rotates the calibrated box axes')
    original = np.asarray(o3d.io.read_point_cloud(str(raw_reference)).points)
    aligned = np.asarray(o3d.io.read_point_cloud(str(aligned_reference)).points)
    require(original.shape == aligned.shape and original.ndim == 2 and original.shape[1] == 3
            and len(original) >= 30, 'Reference point counts or shapes disagree')
    expected_aligned = original @ correction[:3, :3].T + correction[:3, 3]
    require(np.allclose(aligned, expected_aligned, atol=1e-7, rtol=0),
            'Aligned PLY does not match its recorded correction')
    print(f'Frame contract OK: {len(corners)} PnP corners and {len(aligned)} reference points '
          f'in box_center; scan anchor matches the reference export.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('category')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--keypoints', type=Path)
    args = parser.parse_args()
    try:
        verify(args.dataset, args.category, args.calibration, args.keypoints)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f'Frame contract failed: {exc}\n')


if __name__ == '__main__':
    main()
