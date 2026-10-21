#!/usr/bin/env python3
"""Export paired YOLO segmentation and 8-keypoint pose training labels.

The source images, reviewed masks, and rich AprilTag pose labels come from the
turntable dataset workflow.  The two YOLO tasks deliberately use identical
sample stems so segmentation and pose results can be joined at inference time.

Keypoint IDs are the object-frame box corners returned by
``turntable_dataset.box_corners``.  Their immutable order is:

    0 (-X,-Y,-Z)   1 (+X,-Y,-Z)   2 (-X,+Y,-Z)   3 (+X,+Y,-Z)
    4 (-X,-Y,+Z)   5 (+X,-Y,+Z)   6 (-X,+Y,+Z)   7 (+X,+Y,+Z)

Every transform follows ``P_camera = T_camera_object @ P_object``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turntable_dataset import box_corners, project, transform_points  # noqa: E402


KEYPOINT_NAMES = (
    "xneg_yneg_zneg",
    "xpos_yneg_zneg",
    "xneg_ypos_zneg",
    "xpos_ypos_zneg",
    "xneg_yneg_zpos",
    "xpos_yneg_zpos",
    "xneg_ypos_zpos",
    "xpos_ypos_zpos",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read segmentation mask: {path}")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError(f"Segmentation mask must contain only 0 and 255: {path}")
    return mask > 0


def mask_to_polygons(mask, min_area=100, epsilon_ratio=0.001):
    """Convert a mask using objectSegmentation's external-contour policy."""
    binary = np.asarray(mask, dtype=np.uint8) * 255
    if binary.ndim != 2:
        raise ValueError("Segmentation mask must be a 2D image")
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0 if hierarchy is None else int((hierarchy[0, :, 3] >= 0).sum())
    polygons = []
    for index, contour in enumerate(contours):
        if hierarchy[0, index, 3] >= 0 or cv2.contourArea(contour) < min_area:
            continue
        polygon = cv2.approxPolyDP(
            contour, epsilon_ratio * cv2.arcLength(contour, True), True)
        if len(polygon) >= 3 and cv2.contourArea(polygon) > 0:
            polygons.append(polygon[:, 0, :])
    polygons.sort(key=cv2.contourArea, reverse=True)
    return [polygon.astype(np.float64) for polygon in polygons], holes


def mask_polygon(mask, epsilon_ratio=0.001, min_area=100):
    """Return the one object polygon required by this paired pose dataset."""
    polygons, _holes = mask_to_polygons(mask, min_area, epsilon_ratio)
    if len(polygons) != 1:
        raise ValueError(
            f"Expected exactly one object polygon after area filtering; found {len(polygons)}")
    return polygons[0]


def polygon_mask_iou(mask, polygon):
    polygon_mask = np.zeros(np.asarray(mask).shape, dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [np.rint(polygon).astype(np.int32)], 1)
    foreground = np.asarray(mask) > 0
    polygon_foreground = polygon_mask > 0
    intersection = np.count_nonzero(foreground & polygon_foreground)
    union = np.count_nonzero(foreground | polygon_foreground)
    return float(intersection / union) if union else 0.0


def image_sha256(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def mask_bbox(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Cannot calculate a box for an empty mask")
    # Use exclusive maxima so a one-pixel mask still has nonzero width/height.
    return np.asarray([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=np.float64)


def normalize_polygon(points, width, height):
    scale = np.asarray([width, height], dtype=np.float64)
    normalized = np.asarray(points, dtype=np.float64) / scale
    return np.clip(normalized, 0.0, 1.0)


def normalize_bbox(box, width, height):
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float64)
    return np.asarray([
        (x0 + x1) / (2.0 * width),
        (y0 + y1) / (2.0 * height),
        (x1 - x0) / width,
        (y1 - y0) / height,
    ])


def normalize_keypoints(points, width, height):
    """Return eight YOLO (x, y, visibility) keypoints.

    Visibility 2 means that a ground-truth projection is available in the image.
    Off-image projections are encoded as (0, 0, 0), never border-clipped.
    These are virtual cuboid corners, so this flag describes annotation
    availability rather than physical surface visibility.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (8, 2) or not np.all(np.isfinite(points)):
        raise ValueError("box_corners_uv_px must contain eight finite 2D points")
    result = np.zeros((8, 3), dtype=np.float64)
    valid = ((points[:, 0] >= 0) & (points[:, 0] < width)
             & (points[:, 1] >= 0) & (points[:, 1] < height))
    result[valid, :2] = points[valid] / np.asarray([width, height])
    result[valid, 2] = 2
    return result


def number(value):
    return f"{float(value):.8g}"


def segmentation_label(class_id, polygon_normalized):
    fields = [str(int(class_id)), *(number(value) for value in polygon_normalized.ravel())]
    return " ".join(fields) + "\n"


def pose_label(class_id, bbox_normalized, keypoints_normalized):
    fields = [str(int(class_id)), *(number(value) for value in bbox_normalized)]
    for x, y, visibility in keypoints_normalized:
        fields.extend((number(x), number(y), str(int(visibility))))
    return " ".join(fields) + "\n"


def resolve_inputs(dataset, augmentation_root=None, split_manifest=None):
    dataset = Path(dataset).resolve()
    if augmentation_root is None:
        candidates = (dataset / "centerpose_augmented_clean",
                      dataset / "centerpose_augmented")
        augmentation_root = next((path for path in candidates if path.is_dir()), candidates[0])
    augmentation_root = Path(augmentation_root).resolve()
    manifest_path = augmentation_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing reviewed augmentation manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    category = manifest.get("category")
    records = manifest.get("records")
    source_split = manifest.get("split", "train")
    if not category or not isinstance(records, list) or not records:
        raise ValueError("Augmentation manifest needs a category and source records")
    pair_root = augmentation_root / "outf" / f"{category}_{source_split}"

    if split_manifest is None:
        automatic = dataset / "centerpose_training" / "split.json"
        split_manifest = automatic if automatic.is_file() else None
    validation_stems = set()
    split_data = None
    if split_manifest is not None:
        split_manifest = Path(split_manifest).resolve()
        if not split_manifest.is_file():
            raise ValueError(f"Missing split manifest: {split_manifest}")
        split_data = json.loads(split_manifest.read_text())
        if split_data.get("category") != category:
            raise ValueError("Augmentation and split manifest categories differ")
        validation_stems = set(split_data.get("validation_source_stems", []))
        if not validation_stems:
            raise ValueError("Split manifest has no validation source stems")
    return (dataset, augmentation_root, manifest_path, manifest, pair_root,
            split_manifest, split_data, validation_stems)


def sample_plan(dataset, augmentation_root, records, pair_root, validation_stems,
                include_materialized_augmentations=False):
    plan = []
    names = set()
    dimensions = None
    source_stems = {Path(row["source_image"]).stem for row in records}
    missing_validation = validation_stems - source_stems
    if missing_validation:
        raise ValueError(
            f"Validation stems are absent from augmentation manifest: "
            f"{sorted(missing_validation)}")
    for row in records:
        source_stem = Path(row["source_image"]).stem
        label_path = dataset / "labels" / f"{source_stem}.json"
        mask_path = augmentation_root / row.get("mask", f"masks/{source_stem}.png")
        if not label_path.is_file():
            raise ValueError(f"Missing rich pose label: {label_path}")
        if not mask_path.is_file():
            raise ValueError(f"Missing reviewed mask: {mask_path}")
        label = json.loads(label_path.read_text())
        current_dimensions = np.asarray(label.get("box_dimensions_m"), dtype=np.float64)
        if current_dimensions.shape != (3,) or np.any(current_dimensions <= 0):
            raise ValueError(f"Invalid box dimensions in {label_path}")
        if dimensions is None:
            dimensions = current_dimensions
        elif not np.allclose(current_dimensions, dimensions, atol=1e-9, rtol=0):
            raise ValueError("All labels must use one canonical object box revision")
        source_image = Path(row["source_image"]).resolve()
        if not source_image.is_file():
            generated_original = (row.get("generated") or [{}])[0].get("name", "")
            fallback = pair_root / generated_original
            if (Path(generated_original).stem == source_stem and fallback.is_file()):
                source_image = fallback
        if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Missing or unsupported source image: {source_image}")
        original = {"name": source_image.name, "pixel_sha256": None,
                    "image_path": source_image}
        generated = row.get("generated", [])
        if include_materialized_augmentations:
            if (not generated
                    or Path(generated[0].get("name", "")).stem != source_stem
                    or not pair_root.is_dir()):
                raise ValueError(
                    f"Materialized augmentation pairs are missing for {source_stem}")
            choices = generated
        else:
            choices = (original,)
        if source_stem in validation_stems:
            selected = (("val", original),)
        else:
            selected = (("train", item) for item in choices)
        for split, item in selected:
            name = item.get("name", "")
            image_path = Path(item.get("image_path", pair_root / name))
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
                raise ValueError(f"Missing or unsupported augmented image: {image_path}")
            stem = Path(name).stem
            if stem in names:
                raise ValueError(f"Duplicate generated sample stem: {stem}")
            names.add(stem)
            plan.append({
                "split": split,
                "name": name,
                "stem": stem,
                "source_stem": source_stem,
                "image_path": image_path,
                "label_path": label_path,
                "mask_path": mask_path,
                "label": label,
                "original": stem == source_stem,
                "pixel_sha256": item.get("pixel_sha256"),
            })
    if not plan or dimensions is None:
        raise ValueError("No paired samples found")
    return plan, dimensions


def validation_split(records, fraction=0.125, seed=317):
    """Choose held-out source views without relying on materialized augmentations."""
    if not 0 < fraction < 0.5:
        raise ValueError("Validation fraction must be greater than 0 and less than 0.5")
    if len(records) < 4:
        raise ValueError("At least four source views are required for a validation split")
    count = max(1, min(len(records) - 1, int(round(len(records) * fraction))))
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return {Path(records[index]["source_image"]).stem for index in indices[:count]}


def link_or_copy(source, destination, copy_images=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(Path(source).resolve(), destination.parent))


def dataset_yaml(root, class_name, splits, pose=False):
    lines = [f"path: {json.dumps(str(root.resolve()))}"]
    for split in ("train", "val", "test"):
        if split in splits:
            lines.append(f"{split}: images/{split}")
    lines.extend(("names:", f"  0: {json.dumps(class_name)}"))
    if pose:
        lines.extend(("kpt_shape: [8, 3]", "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7]"))
    return "\n".join(lines) + "\n"


def object_keypoints_yaml(dimensions):
    points = box_corners(dimensions)
    lines = [
        "# Canonical object-frame cuboid corners in metres.",
        "# Convention: P_camera = T_camera_object * P_object",
        "object_frame: box_center",
        "units: m",
        "object_keypoints:",
    ]
    for index, (name, xyz) in enumerate(zip(KEYPOINT_NAMES, points)):
        values = ", ".join(number(value) for value in xyz)
        lines.extend((f"  - id: {index}", f"    name: {name}", f"    xyz: [{values}]"))
    lines.extend(("symmetry:", "  type: none"))
    return "\n".join(lines) + "\n"


def preview(image_path, mask, polygon, keypoints, destination):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image for preview: {image_path}")
    overlay = image.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.asarray([40, 180, 40])).astype(np.uint8)
    cv2.polylines(overlay, [np.rint(polygon).astype(np.int32)], True, (0, 255, 0), 2)
    for index, point in enumerate(np.asarray(keypoints)):
        u, v = np.rint(point).astype(int)
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.circle(overlay, (u, v), 5, (0, 80, 255), -1, cv2.LINE_AA)
            cv2.putText(overlay, str(index), (u + 6, v - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), overlay):
        raise OSError(f"Cannot write preview: {destination}")


def export_dataset(dataset, output, augmentation_root=None, split_manifest=None,
                   class_name=None, copy_images=False, epsilon_ratio=0.001,
                   make_previews=True, min_polygon_area=100,
                   min_polygon_mask_iou=0.98,
                   include_materialized_augmentations=False,
                   val_fraction=0.0, split_seed=317):
    (dataset, augmentation_root, manifest_path, manifest, pair_root,
     split_manifest, split_data, validation_stems) = resolve_inputs(
        dataset, augmentation_root, split_manifest)
    class_name = class_name or manifest["category"]
    if split_manifest is None and not validation_stems and val_fraction:
        validation_stems = validation_split(manifest["records"], val_fraction, split_seed)
        split_data = {
            "validation_policy": "deterministic source-view holdout; original images only",
            "validation_fraction": val_fraction,
            "seed": split_seed,
            "validation_source_stems": sorted(validation_stems),
        }
    if (not class_name or class_name != class_name.strip()
            or "\n" in class_name or "\r" in class_name):
        raise ValueError("Class name must be one nonempty line")
    if not 0 <= epsilon_ratio <= 0.05:
        raise ValueError("Polygon epsilon ratio must be between 0 and 0.05")
    if min_polygon_area < 0:
        raise ValueError("Minimum polygon area cannot be negative")
    if not 0 < min_polygon_mask_iou <= 1:
        raise ValueError("Minimum polygon/mask IoU must be in (0, 1]")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output must not already exist: {output}")
    plan, dimensions = sample_plan(
        dataset, augmentation_root, manifest["records"], pair_root, validation_stems,
        include_materialized_augmentations)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    counts = {"train": 0, "val": 0, "test": 0}
    source_cache = {}
    manifest_rows = []
    try:
        for item in plan:
            source_stem = item["source_stem"]
            if source_stem not in source_cache:
                label = item["label"]
                width, height = (int(value) for value in label.get("image_size", ()))
                mask = read_mask(item["mask_path"])
                if mask.shape != (height, width):
                    raise ValueError(
                        f"Mask shape {mask.shape} differs from label image size "
                        f"{(height, width)} for {source_stem}")
                polygons, holes = mask_to_polygons(
                    mask, min_polygon_area, epsilon_ratio)
                if len(polygons) != 1:
                    raise ValueError(
                        f"Expected one object polygon for {source_stem}; "
                        f"found {len(polygons)} after {min_polygon_area:g} px^2 filtering")
                polygon = polygons[0]
                polygon_iou = polygon_mask_iou(mask, polygon)
                if polygon_iou < min_polygon_mask_iou:
                    raise ValueError(
                        f"Polygon/mask disagreement for {source_stem}: IoU "
                        f"{polygon_iou:.4f} < {min_polygon_mask_iou:.4f}")
                bbox = mask_bbox(mask)
                keypoints_px = np.asarray(label.get("box_corners_uv_px"), dtype=np.float64)
                if "intrinsic" in label and "T_camera_object" in label:
                    intrinsic = np.asarray(label["intrinsic"], dtype=np.float64)
                    camera_object = np.asarray(label["T_camera_object"], dtype=np.float64)
                    if intrinsic.shape != (3, 3) or camera_object.shape != (4, 4):
                        raise ValueError(f"Invalid camera geometry in {item['label_path']}")
                    expected = project(
                        intrinsic,
                        transform_points(camera_object, box_corners(dimensions)))
                    if not np.allclose(keypoints_px, expected, atol=1e-4, rtol=0):
                        raise ValueError(
                            f"Stored 2D keypoints do not match canonical box projection: "
                            f"{item['label_path']}")
                source_cache[source_stem] = {
                    "width": width,
                    "height": height,
                    "mask": mask,
                    "polygon_px": polygon,
                    "polygon_normalized": normalize_polygon(polygon, width, height),
                    "polygon_mask_iou": polygon_iou,
                    "polygon_vertices": int(len(polygon)),
                    "holes_omitted": holes,
                    "external_components": len(polygons),
                    "bbox_px": bbox,
                    "bbox_normalized": normalize_bbox(bbox, width, height),
                    "keypoints_px": keypoints_px,
                    "keypoints_normalized": normalize_keypoints(keypoints_px, width, height),
                }
            cached = source_cache[source_stem]
            image = cv2.imread(str(item["image_path"]), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (cached["height"], cached["width"]):
                raise ValueError(f"Image size differs from rich label: {item['image_path']}")
            expected_hash = item.get("pixel_sha256")
            if expected_hash and image_sha256(image) != expected_hash:
                raise ValueError(f"Augmented image changed after manifest creation: "
                                 f"{item['image_path']}")
            split, stem = item["split"], item["stem"]
            for task in ("segmentation", "pose"):
                link_or_copy(item["image_path"], stage / task / "images" / split / item["name"],
                             copy_images)
                link_or_copy(item["mask_path"],
                             stage / task / "masks" / split / f"{stem}.png",
                             copy_images)
            segment_text = segmentation_label(0, cached["polygon_normalized"])
            pose_text = pose_label(0, cached["bbox_normalized"],
                                   cached["keypoints_normalized"])
            segment_path = stage / "segmentation" / "labels" / split / f"{stem}.txt"
            pose_path = stage / "pose" / "labels" / split / f"{stem}.txt"
            segment_path.parent.mkdir(parents=True, exist_ok=True)
            pose_path.parent.mkdir(parents=True, exist_ok=True)
            segment_path.write_text(segment_text)
            pose_path.write_text(pose_text)
            combined = {
                "schema_version": 1,
                "sample": stem,
                "source_sample": source_stem,
                "class_id": 0,
                "class_name": class_name,
                "image_size": [cached["width"], cached["height"]],
                "object_frame": item["label"].get("object_frame", "box_center"),
                "transform_convention": "P_camera = T_camera_object * P_object",
                "segmentation_polygon_xy_px": cached["polygon_px"].tolist(),
                "segmentation_polygon_xy_normalized": cached["polygon_normalized"].tolist(),
                "segmentation_quality": {
                    "polygon_mask_iou": cached["polygon_mask_iou"],
                    "polygon_vertices": cached["polygon_vertices"],
                    "external_components": cached["external_components"],
                    "holes_omitted": cached["holes_omitted"],
                },
                "bbox_xyxy_px": cached["bbox_px"].tolist(),
                "keypoint_order": list(KEYPOINT_NAMES),
                "object_keypoints_xyz_m": box_corners(dimensions).tolist(),
                "keypoints_uv_px": cached["keypoints_px"].tolist(),
                "keypoints_xyv_normalized": cached["keypoints_normalized"].tolist(),
                "T_camera_object": item["label"].get("T_camera_object"),
            }
            write_json(stage / "annotations" / split / f"{stem}.json", combined)
            if make_previews and item["original"]:
                preview(item["image_path"], cached["mask"], cached["polygon_px"],
                        cached["keypoints_px"], stage / "review" / f"{stem}.jpg")
            manifest_rows.append({
                "sample": stem,
                "source_sample": source_stem,
                "split": split,
                "image": str(item["image_path"]),
                "segmentation_label": f"segmentation/labels/{split}/{stem}.txt",
                "pose_label": f"pose/labels/{split}/{stem}.txt",
                "foreground_mask": f"segmentation/masks/{split}/{stem}.png",
                "combined_annotation": f"annotations/{split}/{stem}.json",
            })
            counts[split] += 1

        splits = {name for name, count in counts.items() if count}
        for task in ("segmentation", "pose"):
            (stage / task / "data.yaml").write_text(
                dataset_yaml(output / task, class_name, splits, pose=task == "pose"))
        (stage / "object_keypoints.yaml").write_text(object_keypoints_yaml(dimensions))
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "format": "paired Ultralytics YOLO segmentation and pose datasets with training masks",
            "source_dataset": str(dataset),
            "source_augmentation_manifest": str(manifest_path),
            "source_split_manifest": str(split_manifest) if split_manifest else None,
            "class_name": class_name,
            "class_id": 0,
            "keypoint_count": 8,
            "keypoint_order": list(KEYPOINT_NAMES),
            "keypoint_visibility": (
                "2 = finite ground-truth projection inside image; 0 = outside image; "
                "virtual cuboid corners do not encode physical surface occlusion"),
            "object_frame": "box_center",
            "transform_convention": "P_camera = T_camera_object * P_object",
            "box_dimensions_m": dimensions.tolist(),
            "polygon_epsilon_ratio": epsilon_ratio,
            "minimum_polygon_area_px": min_polygon_area,
            "minimum_polygon_mask_iou": min_polygon_mask_iou,
            "polygon_mask_iou": {
                "minimum": min(value["polygon_mask_iou"] for value in source_cache.values()),
                "mean": float(np.mean([
                    value["polygon_mask_iou"] for value in source_cache.values()])),
            },
            "images": "copied" if copy_images else "relative symlinks",
            "background_augmentation": (
                "materialized legacy images included"
                if include_materialized_augmentations
                else "original images only; masks support training-time background replacement"),
            "counts": {name: count for name, count in counts.items() if count},
            "split_policy": (split_data.get("validation_policy") if split_data
                             else "all generated samples assigned to train"),
            "horizontal_flip_note": (
                "Keypoint IDs are object-frame semantics. Review augmentation policy; "
                "set fliplr=0 unless mirrored samples are intentionally valid."),
            "samples": manifest_rows,
        }
        write_json(stage / "manifest.json", report)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return counts, output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/my_object",
                        help="Turntable dataset containing labels/ (default: datasets/my_object)")
    parser.add_argument("--augmentation-root", type=Path,
                        help="Reviewed mask/augmentation output (default: *_clean, then standard)")
    parser.add_argument("--split-manifest", type=Path,
                        help="Optional source-grouped split.json (auto-detected in the dataset)")
    parser.add_argument("--val-fraction", type=float, default=0.125,
                        help="Source-view validation fraction when no split manifest exists (default: 0.125)")
    parser.add_argument("--split-seed", type=int, default=317,
                        help="Deterministic validation split seed (default: 317)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "datasets/my_object/yolo_seg_pose",
                        help="New output directory; must not already exist")
    parser.add_argument("--class-name", help="YOLO class name (default: augmentation category)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of creating space-saving relative symlinks")
    parser.add_argument("--include-materialized-augmentations", action="store_true",
                        help="Legacy mode: export pre-generated background images from outf/")
    parser.add_argument("--polygon-epsilon", type=float, default=0.001,
                        help="Contour approximation ratio (default: 0.001)")
    parser.add_argument("--min-polygon-area", type=float, default=100,
                        help="Ignore external regions smaller than this pixel area (default: 100)")
    parser.add_argument("--min-polygon-mask-iou", type=float, default=0.98,
                        help="Reject lossy polygon conversion below this IoU (default: 0.98)")
    parser.add_argument("--no-previews", action="store_true",
                        help="Do not render mask and numbered-keypoint review images")
    args = parser.parse_args(argv)
    if not 0 <= args.val_fraction < 0.5:
        parser.error("--val-fraction must be zero or less than 0.5")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        counts, output = export_dataset(
            args.dataset, args.output, args.augmentation_root, args.split_manifest,
            args.class_name, args.copy_images, args.polygon_epsilon,
            not args.no_previews, args.min_polygon_area, args.min_polygon_mask_iou,
            args.include_materialized_augmentations, args.val_fraction,
            args.split_seed)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"YOLO paired-label export failed: {exc}") from exc
    summary = ", ".join(f"{name}={count}" for name, count in counts.items() if count)
    print(f"Done: {output}\nSamples: {summary}\nReview: {output / 'review'}")
    print(f"Segmentation YAML: {output / 'segmentation/data.yaml'}")
    print(f"Pose YAML: {output / 'pose/data.yaml'}")


if __name__ == "__main__":
    main()
