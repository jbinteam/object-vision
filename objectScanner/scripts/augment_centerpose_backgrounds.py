#!/usr/bin/env python3
"""Create reviewed foreground masks; optionally materialize legacy backgrounds."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTATION_PROJECT = ROOT.parent / "objectSegmentation"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def image_sha256(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


class BackgroundReplacer:
    """The random crop/flip compositor used by objectSegmentation."""

    def __init__(self, directory):
        directory = Path(directory).resolve()
        manifest = directory / "sources.json"
        if manifest.is_file():
            self.paths = [directory / row["file"]
                          for row in json.loads(manifest.read_text())["images"]]
        else:
            self.paths = sorted(
                path for path in directory.rglob("*")
                if path.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths or not all(path.is_file() for path in self.paths):
            raise ValueError(f"No usable backgrounds in {directory}")
        self.cache = {}

    def read(self, path):
        path = Path(path)
        if path not in self.cache:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable background: {path}")
            if len(self.cache) >= 16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[path] = image
        return self.cache[path]

    def composite(self, image, mask, rng):
        if mask.shape != image.shape[:2] or not np.any(mask):
            raise ValueError("Composite needs a nonempty, image-sized mask")
        height, width = image.shape[:2]
        path = rng.choice(self.paths)
        background = self.read(path)
        scale = max(width / background.shape[1], height / background.shape[0])
        scale *= rng.uniform(1.0, 1.3)
        resized = cv2.resize(
            background,
            (max(width, int(np.ceil(background.shape[1] * scale))),
             max(height, int(np.ceil(background.shape[0] * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )
        x = rng.randint(0, resized.shape[1] - width)
        y = rng.randint(0, resized.shape[0] - height)
        canvas = resized[y:y + height, x:x + width].copy()
        flipped = rng.random() < 0.5
        if flipped:
            canvas = canvas[:, ::-1].copy()
        # Preserve every selected foreground pixel exactly. The object is never
        # moved, so all CenterPose cuboid labels remain valid without changes.
        canvas[mask > 0] = image[mask > 0]
        return canvas, path, {"crop_xy": [x, y], "scale": scale, "flipped": flipped}


class FullFramePredictor:
    """BiRefNet inference matching objectSegmentation/remove_background.py."""

    def __init__(self, project, device="auto", size=1024):
        import torch
        from PIL import Image
        from torchvision import transforms
        project = Path(project).resolve()
        sys.path.insert(0, str(project / "BiRefNet"))
        from models.birefnet import BiRefNet
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        self.torch = torch
        self.Image = Image
        self.device = device
        self.size = size
        torch.set_num_threads(min(8, torch.get_num_threads()))
        self.model = BiRefNet.from_pretrained(
            str(project / "models/birefnet"), bb_pretrained=False, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def _predict_image(self, image):
        rgb = self.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.transform(rgb).unsqueeze(0).to(self.device)
        context = (self.torch.autocast("cuda", dtype=self.torch.float16)
                   if self.device == "cuda" else nullcontext())
        with self.torch.inference_mode(), context:
            prediction = self.model(inputs)[-1].sigmoid().float()
        prediction = self.torch.nn.functional.interpolate(
            prediction, size=image.shape[:2], mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("BiRefNet returned non-finite foreground probabilities")
        return prediction

    def predict(self, image):
        return self._predict_image(image)

    def predict_crop(self, image, label, margin_ratio=0.6):
        """Run BiRefNet on a pose-guided crop and restore full-frame coordinates."""
        height, width = image.shape[:2]
        box = expected_bbox(label)
        margin = int(round(max(box[2] - box[0], box[3] - box[1]) * margin_ratio))
        x0, y0 = np.floor(box[:2]).astype(int) - margin
        x1, y1 = np.ceil(box[2:]).astype(int) + margin
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Projected pose box produced an empty BiRefNet crop")
        probability = np.zeros((height, width), np.float32)
        probability[y0:y1, x0:x1] = self._predict_image(image[y0:y1, x0:x1])
        return probability


def load_predictor(project, device="auto", size=1024):
    project = Path(project).resolve()
    required = [project / "prepare_emergency_stop.py",
                project / "models/birefnet/model.safetensors",
                project / "BiRefNet/models/birefnet.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("objectSegmentation refinement assets are missing: " + ", ".join(missing))
    return FullFramePredictor(project, device, size)


def expected_bbox(label):
    box = np.asarray(label["bbox_xyxy_px_clipped"], dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Invalid projected bbox in objectScanner label")
    return box


def mask_metrics(mask, label):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Empty refined foreground mask")
    mask_box = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)
    label_box = expected_bbox(label)
    overlap_size = np.maximum(0.0, np.minimum(mask_box[2:], label_box[2:])
                              - np.maximum(mask_box[:2], label_box[:2]))
    intersection = float(np.prod(overlap_size))
    union = (float(np.prod(mask_box[2:] - mask_box[:2]))
             + float(np.prod(label_box[2:] - label_box[:2])) - intersection)
    height, width = mask.shape
    x0, y0 = np.floor(label_box[:2]).astype(int)
    x1, y1 = np.ceil(label_box[2:]).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    within = np.zeros_like(mask, dtype=bool)
    within[y0:y1, x0:x1] = True
    foreground = mask > 0
    center = np.rint(label["center_uv_px"]).astype(int)
    center_inside = (0 <= center[0] < width and 0 <= center[1] < height
                     and bool(foreground[center[1], center[0]]))
    return {
        "mask_area_px": int(np.count_nonzero(foreground)),
        "mask_bbox_xyxy_px": mask_box.astype(int).tolist(),
        "projected_bbox_xyxy_px": label_box.tolist(),
        "bbox_iou": intersection / union if union else 0.0,
        "foreground_inside_projected_bbox": float(
            np.count_nonzero(foreground & within) / np.count_nonzero(foreground)),
        "object_center_inside_mask": center_inside,
    }


def inpaint_exclusion_mask(image, raw_image, minimum_area=20):
    """Recover the regions changed by AprilTag inpainting for mask subtraction."""
    if raw_image is None or raw_image.shape != image.shape:
        raise ValueError("Raw and inpainted images must have matching dimensions")
    changed = (np.max(cv2.absdiff(image, raw_image), axis=2) > 2).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(changed)
    result = np.zeros(image.shape[:2], np.uint8)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < minimum_area:
            continue
        region = (components == component).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, 255, cv2.FILLED)
    return cv2.dilate(result, np.ones((3, 3), np.uint8))


def bright_neutral_disconnect_mask(image):
    """Break connections through pale surfaces or white fiducial paper."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 90)).astype(np.uint8) * 255


def pose_refine_mask(raw_mask, label, kernel_size=11, padding_ratio=0.03,
                     exclusion_mask=None, disconnect_mask=None):
    """Use the calibrated cuboid to isolate the rigid object and break cables."""
    if raw_mask.ndim != 2 or not np.any(raw_mask):
        raise ValueError("BiRefNet produced an empty foreground mask")
    height, width = raw_mask.shape
    box = expected_bbox(label)
    padding = max(2, int(round(max(box[2] - box[0], box[3] - box[1]) * padding_ratio)))
    x0, y0 = np.floor(box[:2]).astype(int) - padding
    x1, y1 = np.ceil(box[2:]).astype(int) + padding
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    candidate = np.zeros_like(raw_mask)
    candidate[y0:y1, x0:x1] = raw_mask[y0:y1, x0:x1]
    if exclusion_mask is not None:
        if exclusion_mask.shape != raw_mask.shape:
            raise ValueError("Exclusion mask dimensions do not match BiRefNet mask")
        candidate[exclusion_mask > 0] = 0
    if disconnect_mask is not None:
        if disconnect_mask.shape != raw_mask.shape:
            raise ValueError("Disconnect mask dimensions do not match BiRefNet mask")
        candidate[disconnect_mask > 0] = 0

    kernel_size = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    count, components, stats, _ = cv2.connectedComponentsWithStats(opened)
    if count < 2:
        raise ValueError("No foreground component remains inside projected 3D box")
    center = np.rint(label["center_uv_px"]).astype(int)
    selected = 0
    if 0 <= center[0] < width and 0 <= center[1] < height:
        selected = int(components[center[1], center[0]])
    if selected == 0:
        selected = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (components == selected).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Selected pose-box foreground has no external contour")
    result = np.zeros_like(raw_mask)
    cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    # Filling the external contour can paint a previously excluded inpainted
    # fiducial back into a concavity. Apply the exact capture-time exclusion a
    # second time so no tag-derived pixels reach a synthetic background.
    if exclusion_mask is not None:
        result[exclusion_mask > 0] = 0
    return result


def refine_probability(probability, label, threshold, min_threshold, exclusion_mask=None,
                       disconnect_mask=None):
    thresholds = np.arange(threshold, min_threshold - 1e-9, -0.1)
    last_error = None
    for selected_threshold in thresholds:
        raw_mask = (probability >= selected_threshold).astype(np.uint8) * 255
        try:
            mask = pose_refine_mask(raw_mask, label, exclusion_mask=exclusion_mask,
                                    disconnect_mask=disconnect_mask)
            break
        except ValueError as error:
            last_error = error
    else:
        raise ValueError(str(last_error))
    return raw_mask, mask, selected_threshold


def refine_mask(image, label, predictor, threshold=0.5, min_threshold=0.2,
                exclusion_mask=None):
    disconnect_mask = bright_neutral_disconnect_mask(image)
    crop_available = hasattr(predictor, "predict_crop")
    inference = "pose_guided_crop" if crop_available else "full_frame"
    probability = (predictor.predict_crop(image, label) if crop_available
                   else predictor.predict(image))
    try:
        raw_mask, mask, selected_threshold = refine_probability(
            probability, label, threshold, min_threshold, exclusion_mask,
            disconnect_mask)
    except ValueError as primary_error:
        if not crop_available:
            raise ValueError(
                f"No usable BiRefNet mask at thresholds {threshold:g} to "
                f"{min_threshold:g}: {primary_error}") from primary_error
        inference = "full_frame_fallback"
        probability = predictor.predict(image)
        try:
            raw_mask, mask, selected_threshold = refine_probability(
                probability, label, threshold, min_threshold, exclusion_mask,
                disconnect_mask)
        except ValueError as fallback_error:
            raise ValueError(
                f"No usable pose-guided crop or full-frame BiRefNet mask at "
                f"thresholds {threshold:g} to {min_threshold:g}: {fallback_error}"
            ) from fallback_error
    metrics = mask_metrics(mask, label)
    metrics.update(birefnet_threshold=float(selected_threshold),
                   birefnet_probability_max=float(probability.max()),
                   birefnet_inference=inference)
    warnings = []
    if metrics["foreground_inside_projected_bbox"] < 0.95:
        warnings.append("foreground extends substantially outside projected 3D box")
    if metrics["bbox_iou"] < 0.2:
        warnings.append("mask bbox and projected 3D bbox have low overlap")
    if not metrics["object_center_inside_mask"]:
        warnings.append("projected object center is outside mask")
    metrics["warnings"] = warnings
    return raw_mask, mask, metrics


def preview_image(image, mask, label, metrics):
    preview = image.copy()
    preview[mask > 0] = (preview[mask > 0] * 0.65
                         + np.asarray([0, 255, 0]) * 0.35).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 0, 255), 2)
    x0, y0, x1, y1 = np.rint(expected_bbox(label)).astype(int)
    cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 180, 0), 2)
    text = (f"IoU {metrics['bbox_iou']:.2f} inside "
            f"{metrics['foreground_inside_projected_bbox']:.2f}")
    cv2.putText(preview, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


def contact_sheets(preview_paths, destination, columns=4, rows=4):
    destination.mkdir(parents=True, exist_ok=True)
    per_page = columns * rows
    for page_index, offset in enumerate(range(0, len(preview_paths), per_page), 1):
        tiles = []
        for path in preview_paths[offset:offset + per_page]:
            image = cv2.imread(str(path))
            tile = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, path.stem[:18], (8, 258), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        tiles += [np.zeros((270, 480, 3), np.uint8)] * (per_page - len(tiles))
        sheet = np.vstack([np.hstack(tiles[index:index + columns])
                           for index in range(0, per_page, columns)])
        write_image(destination / f"sheet_{page_index:02d}.jpg", sheet)


def paired_sources(dataset):
    dataset = Path(dataset).resolve()
    metadata_path = dataset / "centerpose" / "export.json"
    if not metadata_path.is_file():
        raise ValueError(f"CenterPose export metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("split") != "train":
        raise ValueError("Background augmentation is training-only")
    category = metadata["category"]
    pairs = dataset / "centerpose" / "outf" / f"{category}_train"
    images = sorted(path for path in pairs.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    result = []
    for image_path in images:
        centerpose_label = image_path.with_suffix(".json")
        rich_label = dataset / "labels" / f"{image_path.stem}.json"
        if not centerpose_label.is_file() or not rich_label.is_file():
            raise ValueError(f"Missing paired label for {image_path.name}")
        result.append((image_path, centerpose_label, rich_label))
    if not result:
        raise ValueError("No CenterPose training pairs found")
    return metadata, result


def generate(args):
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty or absent: {output}")
    metadata, sources = paired_sources(dataset)
    backgrounds = (BackgroundReplacer(args.backgrounds)
                   if args.augmentations_per_image else None)
    predictor = load_predictor(args.segmentation_project, args.device, args.inference_size)
    rng = random.Random(args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    records = []
    previews = []
    pair_output = stage / "outf" / f"{metadata['category']}_train"
    try:
        for index, (image_path, centerpose_path, rich_path) in enumerate(sources, 1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot read {image_path}")
            label = json.loads(rich_path.read_text())
            raw_path = dataset / label.get("raw_image", f"raw_images/{image_path.name}")
            raw_image = cv2.imread(str(raw_path))
            if raw_image is None:
                raise ValueError(f"Cannot read raw pre-inpainting image: {raw_path}")
            exclusion = inpaint_exclusion_mask(image, raw_image)
            raw_mask, mask, metrics = refine_mask(
                image, label, predictor, args.threshold, args.min_threshold, exclusion)
            if metrics["warnings"] and not args.allow_warnings:
                raise ValueError(f"Mask validation failed for {image_path.name}: "
                                 + "; ".join(metrics["warnings"]))
            stem = image_path.stem
            write_image(stage / "raw_masks" / f"{stem}.png", raw_mask)
            write_image(stage / "masks" / f"{stem}.png", mask)
            if args.write_cutouts:
                write_image(stage / "cutouts" / f"{stem}.png", np.dstack([image, mask]))
            preview_path = stage / "previews" / f"{stem}.jpg"
            write_image(preview_path, preview_image(image, mask, label, metrics))
            previews.append(preview_path)

            generated = [{"name": image_path.name, "background": None,
                          "pixel_sha256": image_sha256(image)}]
            if args.augmentations_per_image:
                pair_output.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, pair_output / image_path.name)
                shutil.copy2(centerpose_path, pair_output / centerpose_path.name)
            for augmentation in range(args.augmentations_per_image):
                composite, background, transform = backgrounds.composite(image, mask, rng)
                name = f"{stem}__bg_{augmentation + 1:03d}.png"
                write_image(pair_output / name, composite)
                shutil.copy2(centerpose_path, pair_output / f"{Path(name).stem}.json")
                generated.append({"name": name,
                                  "background": str(background.resolve()),
                                  "background_transform": transform,
                                  "pixel_sha256": image_sha256(composite)})
            records.append({
                "source_image": str(image_path),
                "source_centerpose_label": str(centerpose_path),
                "source_rich_label": str(rich_path),
                "mask": f"masks/{stem}.png",
                "preview": f"previews/{stem}.jpg",
                "metrics": metrics,
                "generated": generated,
            })
            print(f"[{index}/{len(sources)}] {image_path.name}: "
                  f"{args.augmentations_per_image} augmented", flush=True)

        contact_sheets(previews, stage / "review")
        pair_count = len(list(pair_output.glob("*.png"))) if pair_output.is_dir() else 0
        expected_count = (len(sources) * (args.augmentations_per_image + 1)
                          if args.augmentations_per_image else 0)
        if pair_count != expected_count or len(list(pair_output.glob("*.json"))) != expected_count:
            raise ValueError("Generated CenterPose pair count is inconsistent")
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset),
            "source_centerpose_metadata": metadata,
            "category": metadata["category"],
            "split": "train",
            "source_images": len(sources),
            "augmentations_per_image": args.augmentations_per_image,
            "output_pairs": expected_count,
            "random_seed": args.seed,
            "backgrounds": str(args.backgrounds.resolve()),
            "segmentation_project": str(args.segmentation_project.resolve()),
            "segmentation_method": (
                "pose-guided crop BiRefNet with full-frame fallback; projected-cuboid "
                "constraint; bright-neutral connection breaking; recorded AprilTag-inpaint "
                "region subtraction; 11px cable-breaking opening; projected-center component; "
                "filled external contour"),
            "compositing": ("materialized binary-mask background replacement"
                            if args.augmentations_per_image
                            else "none; masks are consumed by on-the-fly YOLO augmentation"),
            "status": "automatic masks; inspect review sheets before training",
            "records": records,
        }
        write_json(stage / "manifest.json", report)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"Done: {output}\nMaterialized CenterPose pairs: {expected_count}"
          f"\nReview: {output / 'review'}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/my_object")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "datasets/my_object/centerpose_augmented")
    parser.add_argument("--segmentation-project", type=Path,
                        default=DEFAULT_SEGMENTATION_PROJECT)
    parser.add_argument("--backgrounds", type=Path,
                        default=DEFAULT_SEGMENTATION_PROJECT / "backgrounds/picsum")
    parser.add_argument("--augmentations-per-image", type=int, default=0,
                        help="Legacy materialized composites per image; default 0 because YOLO samples backgrounds on the fly")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--inference-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Initial BiRefNet threshold (default: 0.5)")
    parser.add_argument("--min-threshold", type=float, default=0.2,
                        help="Lowest adaptive threshold for pose-box recovery (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-warnings", action="store_true",
                        help="Keep masks that fail geometric review checks")
    parser.add_argument("--write-cutouts", action="store_true",
                        help="Also save RGBA foreground cutouts (not needed for on-the-fly YOLO training)")
    args = parser.parse_args(argv)
    if args.augmentations_per_image < 0:
        parser.error("--augmentations-per-image cannot be negative")
    if args.inference_size < 256 or args.inference_size % 32:
        parser.error("--inference-size must be at least 256 and divisible by 32")
    if not 0 < args.min_threshold <= args.threshold < 1:
        parser.error("Require 0 < min-threshold <= threshold < 1")
    return args


if __name__ == "__main__":
    try:
        generate(parse_args())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"CenterPose augmentation error: {error}")
