#!/usr/bin/env python3
"""Generate OmniNOCS-style dense targets from objectScanner RGB-D captures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nocs_dataset import generate_nocs_target, write_nocs_png  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_gray(path, description):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read {description}: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return image


def relative_symlink(source, destination):
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"Missing source image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def load_inputs(dataset, augmentation_root, split_manifest):
    dataset = Path(dataset).resolve()
    augmentation_root = Path(augmentation_root).resolve()
    augmentation_manifest_path = augmentation_root / "manifest.json"
    if not augmentation_manifest_path.is_file():
        raise ValueError(f"Missing augmentation manifest: {augmentation_manifest_path}")
    augmentation = json.loads(augmentation_manifest_path.read_text())
    records = augmentation.get("records", [])
    category = augmentation.get("category")
    if not category or not records:
        raise ValueError("Augmentation manifest has no category or source records")

    split_manifest = Path(split_manifest).resolve()
    if not split_manifest.is_file():
        raise ValueError(f"Missing source-grouped split manifest: {split_manifest}")
    split = json.loads(split_manifest.read_text())
    if split.get("category") != category:
        raise ValueError("Augmentation and split manifest categories differ")
    validation = set(split.get("validation_source_stems", []))
    return dataset, augmentation_root, augmentation, records, category, split, validation


def preflight(dataset, augmentation_root, records, category):
    pair_root = augmentation_root / "outf" / f"{category}_train"
    resolved = []
    missing_depth = []
    for row in records:
        stem = Path(row["source_image"]).stem
        label_path = dataset / "labels" / f"{stem}.json"
        mask_path = augmentation_root / row.get("mask", f"masks/{stem}.png")
        if not label_path.is_file():
            raise ValueError(f"Missing rich pose label: {label_path}")
        label = json.loads(label_path.read_text())
        depth_path = dataset / label.get("depth", f"depths/{stem}.png")
        if not depth_path.is_file():
            missing_depth.append(str(depth_path))
        if not mask_path.is_file():
            raise ValueError(f"Missing object mask: {mask_path}")
        generated = row.get("generated", [])
        if not generated or Path(generated[0]["name"]).stem != stem:
            raise ValueError(f"Original augmented record is not first for {stem}")
        for item in generated:
            if not (pair_root / item["name"]).is_file():
                raise ValueError(f"Missing augmented RGB image: {pair_root / item['name']}")
        resolved.append((stem, label, depth_path, mask_path, generated))
    return pair_root, resolved, missing_depth


def object_metadata(label, category):
    transform = np.asarray(label["T_camera_object"], dtype=np.float64)
    intrinsic = np.asarray(label["intrinsic"], dtype=np.float64)
    return {
        "objects": [{
            "rotation": transform[:3, :3].tolist(),
            "translation": transform[:3, 3].tolist(),
            "size": list(label["box_dimensions_m"]),
            "object_id": 1,
            "category": category,
        }],
        "intrinsics": {
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
        },
    }


def preview(image_path, mask, rgba, destination):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read preview RGB image: {image_path}")
    valid = rgba[..., 3] > 0
    nocs_rgb = (rgba[..., :3].astype(np.float32) / 257).astype(np.uint8)
    nocs_bgr = cv2.cvtColor(nocs_rgb, cv2.COLOR_RGB2BGR)
    nocs_bgr[~valid] = 0
    overlay = image.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0]
                         + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    canvas = np.hstack((image, overlay, nocs_bgr))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Cannot write NOCS preview: {destination}")


def generate(args):
    dataset, augmentation_root, augmentation, records, category, split, validation = load_inputs(
        args.dataset, args.augmentation_root, args.split_manifest)
    pair_root, sources, missing_depth = preflight(
        dataset, augmentation_root, records, category)
    readiness = {
        "source_views": len(sources),
        "depth_images_found": len(sources) - len(missing_depth),
        "depth_images_missing": len(missing_depth),
        "missing_depth_examples": missing_depth[:10],
        "ready": not missing_depth,
    }
    if args.check_only:
        print(json.dumps(readiness, indent=2))
        return readiness
    if missing_depth:
        raise ValueError(
            f"Aligned depth is missing for {len(missing_depth)}/{len(sources)} source views. "
            "The existing RGB-only captures cannot produce dense NOCS targets; capture new "
            "samples with the updated collector or supply an aligned-mesh renderer.")

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty or absent: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    metadata = {"train": [], "val": []}
    source_metrics = []
    try:
        for index, (stem, label, depth_path, mask_path, generated) in enumerate(sources, 1):
            depth = read_gray(depth_path, "aligned depth")
            mask = read_gray(mask_path, "object mask")
            instance, rgba, metrics = generate_nocs_target(
                depth, mask, label, args.box_tolerance_ratio)
            if metrics["valid_nocs_pixels"] < args.min_valid_pixels:
                raise ValueError(
                    f"{stem} has only {metrics['valid_nocs_pixels']} valid NOCS pixels; "
                    f"minimum is {args.min_valid_pixels}")
            if metrics["nocs_coverage"] < args.min_nocs_coverage:
                raise ValueError(
                    f"{stem} NOCS coverage {metrics['nocs_coverage']:.3f} is below "
                    f"{args.min_nocs_coverage:.3f}")
            prefix = stage / "annotations" / stem
            prefix.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(prefix) + "_instances.png", instance):
                raise OSError(f"Cannot write instance PNG for {stem}")
            write_nocs_png(str(prefix) + "_nocs.png", rgba)
            preview(pair_root / generated[0]["name"], mask, rgba,
                    stage / "review" / f"{stem}.jpg")

            target_split = "val" if stem in validation else "train"
            selected = generated[:1] if target_split == "val" else generated
            common = object_metadata(label, category)
            for item in selected:
                name = item["name"]
                relative_image = Path("images") / target_split / name
                relative_symlink(pair_root / name, stage / relative_image)
                metadata[target_split].append({
                    **common,
                    "image_name": relative_image.as_posix(),
                    "omninocs_name": f"annotations/{stem}",
                    "nocs_image_downscale": 1.0,
                    "source_view": stem,
                    "is_background_augmented": item.get("background") is not None,
                })
            source_metrics.append({"source_view": stem, **metrics})
            print(f"[{index}/{len(sources)}] {stem}: {metrics['valid_nocs_pixels']} valid pixels",
                  flush=True)

        for name in ("train", "val"):
            write_json(stage / f"{category}_{name}_metadata.json", metadata[name])
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "format": "OmniNOCS-compatible uint16 instance and XYZ-valid RGBA PNG",
            "source_dataset": str(dataset),
            "source_augmentation_manifest": str(augmentation_root / "manifest.json"),
            "source_split_manifest": str(Path(args.split_manifest).resolve()),
            "category": category,
            "coordinate_frame": "box_center; right-handed; source objectScanner Z-up",
            "coordinate_normalization": "p_object / norm(box_dimensions_m)",
            "png_packing": "XYZ = (normalized + 0.5) * 65535; A = valid * 65535",
            "depth_encoding": "uint16 millimetres; 0 invalid; aligned to rectified color",
            "instance_encoding": "uint16; background=0; object=1",
            "split_policy": split.get("validation_policy"),
            "source_views": len(sources),
            "images": {name: len(rows) for name, rows in metadata.items()},
            "box_tolerance_ratio": args.box_tolerance_ratio,
            "minimum_valid_pixels": args.min_valid_pixels,
            "minimum_nocs_coverage": args.min_nocs_coverage,
            "source_metrics": source_metrics,
        }
        write_json(stage / "manifest.json", manifest)
        if output.exists():
            output.rmdir()
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"Done: {output}\nTrain images: {len(metadata['train'])}\n"
          f"Validation images: {len(metadata['val'])}\nReview: {output / 'review'}")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "datasets/my_object_contrast")
    parser.add_argument("--augmentation-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--box-tolerance-ratio", type=float, default=0.08)
    parser.add_argument("--min-valid-pixels", type=int, default=256)
    parser.add_argument("--min-nocs-coverage", type=float, default=0.15)
    parser.add_argument("--check-only", action="store_true",
                        help="Report missing inputs without writing a dataset")
    args = parser.parse_args(argv)
    if args.augmentation_root is None:
        args.augmentation_root = args.dataset / "centerpose_augmented"
    if args.split_manifest is None:
        args.split_manifest = args.dataset / "centerpose_training/split.json"
    if args.output is None:
        args.output = args.dataset / "omninocs"
    if not 0 <= args.box_tolerance_ratio <= 0.5:
        parser.error("--box-tolerance-ratio must be between 0 and 0.5")
    if args.min_valid_pixels < 1:
        parser.error("--min-valid-pixels must be positive")
    if not 0 < args.min_nocs_coverage <= 1:
        parser.error("--min-nocs-coverage must be in (0, 1]")
    return args


if __name__ == "__main__":
    try:
        generate(parse_args())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"NOCS dataset error: {error}")
