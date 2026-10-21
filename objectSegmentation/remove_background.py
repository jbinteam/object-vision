"""Batch foreground extraction, with optional draft YOLO segmentation labels."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def mask_to_polygons(mask, min_area=100, epsilon=0.001):
    """Return external contours only; disconnected regions are separate rows."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0 if hierarchy is None else int((hierarchy[0, :, 3] >= 0).sum())
    polygons = []
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] >= 0 or cv2.contourArea(contour) < min_area:
            continue
        polygon = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
        if len(polygon) >= 3 and cv2.contourArea(polygon) > 0:
            polygons.append(polygon.reshape(-1, 2))
    polygons.sort(key=cv2.contourArea, reverse=True)
    return polygons, holes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image or directory (searched recursively)")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "birefnet")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--size", type=int, default=1024, help="Inference size, multiple of 32")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=float, default=100, help="Minimum contour area in original pixels")
    parser.add_argument("--epsilon", type=float, default=0.001, help="Polygon simplification fraction of perimeter")
    parser.add_argument("--class-name", help="Enable single-class YOLO export (class ID 0)")
    args = parser.parse_args()
    if args.size < 256 or args.size % 32:
        parser.error("--size must be at least 256 and a multiple of 32")
    if not 0 < args.threshold < 1 or args.min_area < 0 or not 0 <= args.epsilon <= 0.05:
        parser.error("Require 0 < threshold < 1, min-area >= 0, and 0 <= epsilon <= 0.05")
    if args.class_name is not None and not args.class_name.strip():
        parser.error("--class-name must not be empty")
    return args


def main():
    args = parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if not source.exists():
        raise SystemExit(f"Input does not exist: {source}")
    if output == source or (source.is_dir() and output.is_relative_to(source)):
        raise SystemExit("Output must be outside the input directory.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty; choose a new directory: {output}")
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    files = [p for p in files if p.suffix.lower() in EXTENSIONS]
    if not files:
        raise SystemExit("No supported images found.")
    if not (args.model / "model.safetensors").is_file():
        raise SystemExit("Weights missing. Run: .venv/bin/python download_model.py")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu or check the PyTorch installation.")
    # Use the cloned upstream implementation, without executing Hugging Face remote code.
    sys.path.insert(0, str(ROOT / "BiRefNet"))
    from models.birefnet import BiRefNet

    torch.set_num_threads(min(8, torch.get_num_threads()))
    model = BiRefNet.from_pretrained(str(args.model.resolve()), bb_pretrained=False, strict=True)
    model.eval().requires_grad_(False).to(device)
    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    folders = ["alpha", "masks", "cutouts", "previews"]
    if args.class_name is not None:
        folders += ["images", "labels"]
    for folder in folders:
        (output / folder).mkdir(parents=True, exist_ok=True)
    records = []
    print(f"Processing {len(files)} image(s) on {device} at {args.size}px", flush=True)
    start = time.perf_counter()
    for index, path in enumerate(files, start=1):
        # Keep the original extension in the key to avoid foo.jpg/foo.png collisions.
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        key = relative.parent / relative.name
        with Image.open(path) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
        inputs = transform(rgb).unsqueeze(0).to(device)
        context = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
        with torch.inference_mode(), context:
            prediction = model(inputs)[-1].sigmoid().float()
        prediction = torch.nn.functional.interpolate(
            prediction, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite prediction for {path}")
        alpha = Image.fromarray(np.rint(prediction * 255).astype(np.uint8))
        mask = (prediction >= args.threshold).astype(np.uint8) * 255
        polygons, holes = mask_to_polygons(mask, args.min_area, args.epsilon)
        cutout = rgb.copy()
        cutout.putalpha(alpha)
        preview = np.array(rgb)
        preview[mask > 0] = (0.65 * preview[mask > 0] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.polylines(preview, polygons, True, (255, 0, 0), 2)
        products = {"alpha": alpha, "masks": Image.fromarray(mask), "cutouts": cutout,
                    "previews": Image.fromarray(preview)}
        if args.class_name is not None:
            products["images"] = rgb
        for folder, product in products.items():
            destination = output / folder / f"{key}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            product.save(destination)
        if args.class_name is not None:
            rows = []
            for polygon in polygons:
                normalized = polygon / np.array([rgb.width, rgb.height])
                rows.append("0 " + " ".join(f"{v:.6f}" for v in normalized.ravel()))
            label = output / "labels" / f"{key}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(rows) + ("\n" if rows else ""))
        warnings = []
        if not polygons:
            warnings.append("no usable foreground polygon")
        if len(polygons) > 1:
            warnings.append("multiple regions: verify object instances")
        if holes:
            warnings.append("holes omitted from YOLO polygons; retained in masks")
        records.append({"source": str(path), "key": str(key), "width": rgb.width,
                        "height": rgb.height, "polygons": len(polygons), "holes": holes,
                        "foreground_fraction": float((mask > 0).mean()), "warnings": warnings})
        print(f"[{index}/{len(files)}] {relative}: {len(polygons)} region(s)" +
              ("; " + "; ".join(warnings) if warnings else ""), flush=True)
    report = {"model": str(args.model.resolve()), "device": device, "size": args.size,
              "threshold": args.threshold, "min_area": args.min_area, "epsilon": args.epsilon,
              "class_names": [args.class_name] if args.class_name is not None else [],
              "elapsed_seconds": time.perf_counter() - start, "images": records}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
