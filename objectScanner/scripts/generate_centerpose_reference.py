#!/usr/bin/env python3
"""Build observed clouds and a fused reference in the calibrated box-center frame.

Inputs are depth-enabled turntable captures plus their reviewed segmentation
masks.  The saved pose convention is always:

    P_camera = T_camera_object @ P_object

The fused ``reference.ply`` is already in ``object_frame == box_center``. Its
origin is the CenterPose cuboid center and its basis is the same basis used by
the 3D/2D keypoints, PnP, GICP, and TF. Do not transform it again at runtime.
"""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nocs_dataset import extract_masked_object_points  # noqa: E402
from turntable_dataset import box_corners  # noqa: E402


KEYPOINT_NAMES = (
    "xneg_yneg_zneg", "xpos_yneg_zneg", "xneg_ypos_zneg", "xpos_ypos_zneg",
    "xneg_yneg_zpos", "xpos_yneg_zpos", "xneg_ypos_zpos", "xpos_ypos_zpos",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def resolve_augmentation(dataset, augmentation_root=None):
    dataset = Path(dataset).resolve()
    if not dataset.is_dir():
        raise ValueError(
            f"Dataset directory does not exist: {dataset}. Capture a real session first; "
            "example names such as NEW_CAPTURE are placeholders.")
    if augmentation_root is None:
        candidates = (dataset / "centerpose_augmented_clean",
                      dataset / "centerpose_augmented")
        augmentation_root = next((path for path in candidates if path.is_dir()), candidates[0])
    augmentation_root = Path(augmentation_root).resolve()
    manifest_path = augmentation_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing reviewed augmentation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records", [])
    if not records:
        raise ValueError("Augmentation manifest contains no source records")
    return dataset, augmentation_root, manifest_path, manifest, records


def preflight(dataset, augmentation_root, records):
    resolved = []
    missing = {"labels": [], "images": [], "depths": [], "masks": []}
    dimensions = None
    for row in records:
        stem = Path(row["source_image"]).stem
        label_path = dataset / "labels" / f"{stem}.json"
        if not label_path.is_file():
            missing["labels"].append(str(label_path))
            continue
        label = json.loads(label_path.read_text())
        image_path = dataset / label.get("image", f"images/{stem}.png")
        depth_path = dataset / label.get("depth", f"depths/{stem}.png")
        mask_path = augmentation_root / row.get("mask", f"masks/{stem}.png")
        for key, path in (("images", image_path), ("depths", depth_path),
                          ("masks", mask_path)):
            if not path.is_file():
                missing[key].append(str(path))
        current_dimensions = np.asarray(label.get("box_dimensions_m"), dtype=np.float64)
        if current_dimensions.shape != (3,) or np.any(current_dimensions <= 0):
            raise ValueError(f"Invalid box_dimensions_m in {label_path}")
        if dimensions is None:
            dimensions = current_dimensions
        elif not np.allclose(current_dimensions, dimensions, atol=1e-9, rtol=0):
            raise ValueError("All source views must use one canonical object-box revision")
        resolved.append((stem, label_path, label, image_path, depth_path, mask_path))
    report = {
        "source_views": len(records),
        "ready_views": sum(
            label_path.is_file() and image_path.is_file() and depth_path.is_file()
            and mask_path.is_file()
            for _stem, label_path, _label, image_path, depth_path, mask_path in resolved),
        "missing_counts": {key: len(values) for key, values in missing.items() if values},
        "missing_examples": {key: values[:5] for key, values in missing.items() if values},
        "ready": not any(missing.values()) and len(resolved) == len(records),
    }
    return resolved, dimensions, report


def read_inputs(item):
    stem, _label_path, label, image_path, depth_path, mask_path = item
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or depth is None or mask is None:
        raise ValueError(f"Cannot decode RGB, depth, or mask for {stem}")
    if image.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise ValueError(f"RGB/depth/mask dimensions differ for {stem}")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Depth must be a single-channel uint16 PNG for {stem}")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError(f"Mask must be binary for {stem}")
    return image, depth, mask


def make_cloud(o3d, points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud


def save_cloud(o3d, path, cloud):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(cloud.points) < 3 or not o3d.io.write_point_cloud(
            str(path), cloud, write_ascii=False, compressed=False):
        raise OSError(f"Cannot write point cloud: {path}")


def orient_normals_from_center(cloud):
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    if not len(points) or normals.shape != points.shape or not np.isfinite(normals).all():
        raise ValueError("Reference normal estimation failed")
    inward = np.einsum("ij,ij->i", normals, points) < 0
    normals[inward] *= -1


def object_keypoints_yaml(dimensions):
    lines = [
        "# The reference cloud and these keypoints share this frame exactly.",
        "# Convention: P_camera = T_camera_object * P_object",
        "object_frame: box_center",
        "units: m",
        "object_keypoints:",
    ]
    for index, (name, xyz) in enumerate(zip(KEYPOINT_NAMES, box_corners(dimensions))):
        values = ", ".join(f"{float(value):.8g}" for value in xyz)
        lines.extend((f"  - id: {index}", f"    name: {name}",
                      f"    xyz: [{values}]"))
    lines.extend(("symmetry:", "  type: none"))
    return "\n".join(lines) + "\n"


def generate(args):
    import open3d as o3d

    dataset, augmentation_root, manifest_path, manifest, records = resolve_augmentation(
        args.dataset, args.augmentation_root)
    items, dimensions, check = preflight(dataset, augmentation_root, records)
    if args.check_only:
        print(json.dumps(check, indent=2))
        return check
    if not check["ready"]:
        examples = [path for values in check["missing_examples"].values()
                    for path in values][:5]
        raise ValueError(
            f"Reference inputs are incomplete ({check['ready_views']}/{check['source_views']} "
            f"views ready). Examples: {examples}")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"Output must not already exist: {output}")
    if args.voxel_size_mm < 0 or args.observation_voxel_size_mm < 0:
        raise ValueError("Voxel sizes cannot be negative")
    if args.min_valid_points < 3:
        raise ValueError("Minimum valid points must be at least 3")
    if not 0 <= args.box_tolerance_ratio <= 0.5:
        raise ValueError("Box tolerance ratio must be between 0 and 0.5")
    if args.normal_radius_mm <= 0 or args.normal_max_neighbors < 3:
        raise ValueError("Normal parameters must be positive")
    if args.outlier_neighbors < 0 or args.outlier_std_ratio <= 0:
        raise ValueError("Outlier parameters are invalid")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    all_points, all_colors, view_rows = [], [], []
    try:
        for index, item in enumerate(items, 1):
            stem, _label_path, label, _image_path, _depth_path, _mask_path = item
            image, depth, mask = read_inputs(item)
            (_foreground, ys, xs, camera_points,
             object_points, metrics) = extract_masked_object_points(
                depth, mask, label, args.box_tolerance_ratio)
            if len(object_points) < args.min_valid_points:
                raise ValueError(
                    f"{stem} has only {len(object_points)} valid object points; "
                    f"need {args.min_valid_points}")
            colors = image[ys, xs, ::-1].astype(np.float64) / 255.0
            camera_cloud = make_cloud(o3d, camera_points, colors)
            object_cloud = make_cloud(o3d, object_points, colors)
            if args.observation_voxel_size_mm:
                voxel = args.observation_voxel_size_mm / 1000.0
                camera_cloud = camera_cloud.voxel_down_sample(voxel)
                object_cloud = object_cloud.voxel_down_sample(voxel)
            if not args.no_observations:
                save_cloud(o3d, stage / "observations/camera_frame" / f"{stem}.ply",
                           camera_cloud)
                save_cloud(o3d, stage / "observations/object_frame" / f"{stem}.ply",
                           object_cloud)
            object_array = np.asarray(object_cloud.points).copy()
            color_array = np.asarray(object_cloud.colors).copy()
            all_points.append(object_array)
            all_colors.append(color_array)
            view_rows.append({
                "source_view": stem,
                "camera_frame": label.get("camera_frame", "color_optical_frame"),
                "object_frame": label.get("object_frame", "box_center"),
                "T_camera_object": label["T_camera_object"],
                "raw_valid_points": int(len(object_points)),
                "saved_points": int(len(object_array)),
                "metrics": metrics,
            })
            print(f"[{index}/{len(items)}] {stem}: {len(object_points)} valid points",
                  flush=True)

        fused = make_cloud(o3d, np.vstack(all_points), np.vstack(all_colors))
        points_before = len(fused.points)
        removed_outliers = 0
        if args.outlier_neighbors:
            filtered, kept = fused.remove_statistical_outlier(
                nb_neighbors=args.outlier_neighbors, std_ratio=args.outlier_std_ratio)
            removed_outliers = len(fused.points) - len(kept)
            fused = filtered
        points_after_outlier = len(fused.points)
        if args.voxel_size_mm:
            fused = fused.voxel_down_sample(args.voxel_size_mm / 1000.0)
        if len(fused.points) < args.min_valid_points:
            raise ValueError("Reference has too few points after filtering/downsampling")
        fused.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.normal_radius_mm / 1000.0,
            max_nn=args.normal_max_neighbors))
        orient_normals_from_center(fused)
        save_cloud(o3d, stage / "reference.ply", fused)

        identity = np.eye(4).tolist()
        metadata = {
            "schema_version": 3,
            "name": output.name,
            "frame_id": "box_center",
            "object_frame": "box_center",
            "units": "m",
            "origin_definition": "CenterPose cuboid center; [0, 0, 0] in reference.ply",
            "axis_definition": (
                "Identity object basis from turntable calibration; shared by 3D keypoints, "
                "2D projections, PnP, GICP, and TF"),
            "transform_convention": "P_camera = T_camera_object * P_object",
            "reference_already_in_object_frame": True,
            # Compatibility with the existing reference metadata reader. There
            # is no separate scan frame because fusion happened in object_frame.
            "T_scan_obj": identity,
            "T_obj_scan": identity,
            "box_dimensions_m": dimensions.tolist(),
            "source": {
                "dataset": str(dataset),
                "augmentation_manifest": str(manifest_path),
                "views": len(items),
                "category": manifest.get("category"),
            },
            "points_before_filtering": points_before,
            "statistical_outliers_removed": removed_outliers,
            "points_after_outlier_filtering": points_after_outlier,
            "points": len(fused.points),
            "has_colors": fused.has_colors(),
            "has_normals": fused.has_normals(),
            "voxel_size_mm": args.voxel_size_mm,
            "observation_voxel_size_mm": args.observation_voxel_size_mm,
            "box_tolerance_ratio": args.box_tolerance_ratio,
            "normal_radius_mm": args.normal_radius_mm,
            "normals_orientation": "outward from box_center where locally defined",
            "views": view_rows,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(stage / "frame.json", metadata)
        (stage / "object_keypoints.yaml").write_text(object_keypoints_yaml(dimensions))
        (stage / "README.txt").write_text(
            "reference.ply is fused from reviewed masked RGB-D observations.\n"
            "It is already expressed in box_center, in metres.\n"
            "The origin is the CenterPose cuboid center and the basis is the calibrated object basis.\n"
            "Use it directly as the GICP source; do not apply another frame transform.\n"
            "Each saved camera observation maps to object coordinates with inverse(T_camera_object).\n"
            "Inspect the fused cloud for mask leakage, missing views, and depth artifacts before training/tracking.\n")
        with zipfile.ZipFile(stage / "reference_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
            for filename in ("reference.ply", "frame.json", "object_keypoints.yaml",
                             "README.txt"):
                bundle.write(stage / filename, arcname=f"{output.name}/{filename}")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"Done: {output}\nReference points: {len(fused.points)}\n"
          f"Frame: box_center (CenterPose origin)")
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/my_object")
    parser.add_argument("--augmentation-root", type=Path,
                        help="Reviewed masks (default: centerpose_augmented_clean, then standard)")
    parser.add_argument("--output", type=Path,
                        help="New reference directory (default: references/DATASET_centerpose)")
    parser.add_argument("--box-tolerance-ratio", type=float, default=0.08)
    parser.add_argument("--min-valid-points", type=int, default=100)
    parser.add_argument("--observation-voxel-size-mm", type=float, default=0,
                        help="Optional per-view downsampling before saving/fusion")
    parser.add_argument("--voxel-size-mm", type=float, default=2.0,
                        help="Final fused-reference voxel size (default: 2 mm)")
    parser.add_argument("--outlier-neighbors", type=int, default=20,
                        help="Statistical outlier neighbors; 0 disables (default: 20)")
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--normal-radius-mm", type=float, default=10.0)
    parser.add_argument("--normal-max-neighbors", type=int, default=30)
    parser.add_argument("--no-observations", action="store_true",
                        help="Write only the fused reference, not per-view PLY files")
    parser.add_argument("--check-only", action="store_true",
                        help="Report whether RGB/depth/mask/pose inputs are complete")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output is None:
        args.output = ROOT / "references" / f"{args.dataset.name}_centerpose"
    try:
        generate(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"CenterPose reference generation failed: {exc}") from exc


if __name__ == "__main__":
    main()
