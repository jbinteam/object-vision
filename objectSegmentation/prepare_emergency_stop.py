"""Refine this L515 emergency-stop capture into cable-free draft YOLO labels.

Uses the yellow enclosure to select the object from BiRefNet masks. This is a
dataset-specific cleanup, not a general-purpose semantic segmentation model.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def yellow_seed(image, return_candidates=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([12, 85, 45]), np.array([40, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError("No yellow enclosure found")
    # Wood can be orange/yellow too. Select the enclosure surrounding a solid
    # red button, rather than whichever yellow region happens to be largest.
    red = cv2.inRange(hsv, np.array([0, 80, 40]), np.array([10, 255, 255]))
    red |= cv2.inRange(hsv, np.array([170, 80, 40]), np.array([179, 255, 255]))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    candidates = []
    for candidate in range(1, count):
        x, y, width, height, area = stats[candidate]
        if area >= 1000:
            # At a low viewing angle the button projects above the yellow box.
            # Include that space while retaining the enclosure's horizontal span.
            candidates.append((int(np.count_nonzero(red[max(0, y-height//2):y+height, x:x+width])), candidate))
    if not candidates or max(candidates)[0] < 500:
        raise ValueError("No reliable yellow enclosure surrounding a red button found")
    _, index = max(candidates)
    result = ((labels == index).astype(np.uint8) * 255, stats[index, :4])
    if return_candidates:
        return (*result, sum(score >= 500 for score, _ in candidates))
    return result


def clean_mask(mask, seed, kernel_size=17):
    """Break thin cable connections, select enclosure, and fill interior holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, _, _ = cv2.connectedComponentsWithStats(opened)
    scores = np.bincount(labels[seed > 0], minlength=count)
    scores[0] = 0
    best = int(np.argmax(scores))
    selected = (labels == best).astype(np.uint8) * 255 if best else np.zeros_like(mask)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if contours:
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    coverage = np.count_nonzero((result > 0) & (seed > 0)) / np.count_nonzero(seed)
    return result, coverage


def exclude_red_wire(mask, image, box):
    """Remove red regions connected to cable outside the enclosure footprint."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 80, 35]), np.array([5, 255, 255]))
    red |= cv2.inRange(hsv, np.array([175, 80, 35]), np.array([179, 255, 255]))
    seed, _ = yellow_seed(image)
    body, _ = clean_mask(mask, seed)
    points = cv2.findNonZero(body)
    if points is None:
        return mask.copy()
    x, y, width, height = cv2.boundingRect(points)
    outside = np.ones(mask.shape, np.uint8)
    outside[max(0, y-8):y+height+8, max(0, x-8):x+width+8] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red)
    exterior_counts = np.bincount(labels[outside > 0], minlength=count)
    candidates = np.flatnonzero((exterior_counts >= 20) & (stats[:, cv2.CC_STAT_AREA] >= 40))
    candidates = candidates[candidates != 0]
    wire = np.isin(labels, candidates).astype(np.uint8)
    wire = cv2.dilate(wire, np.ones((3, 3), np.uint8))
    result = mask.copy()
    result[wire > 0] = 0
    return result


class CropPredictor:
    def __init__(self):
        import torch
        from torchvision import transforms
        sys.path.insert(0, str(ROOT / "BiRefNet"))
        from models.birefnet import BiRefNet
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(min(8, torch.get_num_threads()))
        self.model = BiRefNet.from_pretrained(str(ROOT / "models/birefnet"), bb_pretrained=False, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image, box):
        from contextlib import nullcontext
        from PIL import Image
        x, y, width, height = map(int, box)
        margin = int(max(width, height) * 0.6)
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(image.shape[1], x + width + margin), min(image.shape[0], y + height + margin)
        crop = image[y0:y1, x0:x1]
        torch = self.torch
        tensor = self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        context = torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()
        with torch.inference_mode(), context:
            prediction = self.model(tensor)[-1].sigmoid().float()
        prediction = torch.nn.functional.interpolate(prediction, size=crop.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite crop prediction")
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[y0:y1, x0:x1] = (prediction >= 0.5).astype(np.uint8) * 255
        return mask


def write_image(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Cannot write {path}")


def prepare_image(source, key, out, predictor):
    """Generate one draft annotation directly from RGB, without a raw-mask pass."""
    source, key, out = Path(source), Path(key), Path(out)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Cannot read {source}")
    seed, box, candidate_count = yellow_seed(image, return_candidates=True)
    raw_mask = predictor.predict(image, box)
    mask, coverage = clean_mask(exclude_red_wire(raw_mask, image, box), seed)
    if coverage < 0.9:
        raise ValueError(f"Insufficient enclosure coverage: {coverage:.3f}")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if len(polygon) < 3:
        raise ValueError("Invalid polygon")
    normalized = polygon / np.array([image.shape[1], image.shape[0]])
    destination = out / "images" / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    label = (out / "labels" / key).with_suffix(".txt")
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 " + " ".join(f"{v:.6f}" for v in normalized.ravel()) + "\n")
    write_image(out / "crop_masks" / key, raw_mask)
    write_image(out / "masks" / key, mask)
    write_image(out / "cutouts" / key, np.dstack([image, mask]))
    preview = image.copy()
    preview[mask > 0] = (preview[mask > 0] * .65 + np.array([0, 255, 0]) * .35).astype(np.uint8)
    cv2.polylines(preview, [polygon], True, (0, 0, 255), 2)
    write_image(out / "previews" / key, preview)
    return {"source": str(source.resolve()), "key": str(key), "class_id": 0,
            "class_name": "emergency_stop", "crop_inference": True,
            "review_flags": ["additional emergency-stop candidate outside primary mask"] if candidate_count > 1 else [],
            "seed_coverage": coverage, "mask_area": int(np.count_nonzero(mask)),
            "polygon_vertices": len(polygon), "bbox_xywh": list(cv2.boundingRect(contour))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "output/l515_100")
    parser.add_argument("--output", type=Path, default=ROOT / "output/l515_emergency_stop")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Choose a new or empty output directory")
    raw = json.loads((args.raw / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    records, tiles, full_tiles = [], [], []
    predictor = None
    for index, row in enumerate(raw["images"], 1):
        key = Path(row["key"])
        image = cv2.imread(row["source"])
        if image is None:
            raise ValueError(f"Cannot read {row['source']}")
        seed, box, candidate_count = yellow_seed(image, return_candidates=True)
        original = cv2.imread(str(args.raw / "masks" / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"Missing raw mask: {key}")
        _, original_coverage = clean_mask(original, seed)
        recovered = original_coverage < 0.9
        if predictor is None:
            predictor = CropPredictor()
        original = predictor.predict(image, box)
        write_image(out / "crop_masks" / key, original)
        original = exclude_red_wire(original, image, box)
        mask, coverage = clean_mask(original, seed)
        if coverage < 0.9:
            raise ValueError(f"Insufficient enclosure coverage after crop retry: {key} ({coverage:.3f})")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(polygon) < 3:
            raise ValueError(f"Invalid polygon: {key}")
        normalized = polygon / np.array([image.shape[1], image.shape[0]])
        destination = out / "images" / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source"], destination)
        label = (out / "labels" / key).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 " + " ".join(f"{v:.6f}" for v in normalized.ravel()) + "\n")
        write_image(out / "masks" / key, mask)
        write_image(out / "cutouts" / key, np.dstack([image, mask]))
        preview = image.copy()
        preview[mask > 0] = (preview[mask > 0] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
        cv2.polylines(preview, [polygon], True, (0, 0, 255), 2)
        write_image(out / "previews" / key, preview)
        full_tile = cv2.resize(preview, (480, 270))
        cv2.putText(full_tile, f"{index:03d}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        full_tiles.append(full_tile)
        x, y, w, h = cv2.boundingRect(contour)
        pad = 50
        tile = preview[max(0, y-pad):min(image.shape[0], y+h+pad), max(0, x-pad):min(image.shape[1], x+w+pad)]
        scale = min(350/tile.shape[1], 260/tile.shape[0])
        tile = cv2.resize(tile, (round(tile.shape[1]*scale), round(tile.shape[0]*scale)))
        canvas = np.full((300, 360, 3), 32, np.uint8)
        canvas[30:30+tile.shape[0], :tile.shape[1]] = tile
        title = f"{index:03d} session {key.parts[0][-8:]}" + (" RETRY" if recovered else "")
        cv2.putText(canvas, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
        tiles.append(canvas)
        records.append({"index": index, "source": row["source"], "key": str(key),
                        "class_id": 0, "class_name": "emergency_stop", "crop_inference": True,
                        "recovered_raw_miss": recovered,
                        "review_flags": ["additional emergency-stop candidate outside primary mask"] if candidate_count > 1 else [],
                        "seed_coverage": coverage, "mask_area": int(np.count_nonzero(mask)),
                        "polygon_vertices": len(polygon), "bbox_xywh": [x, y, w, h]})
        print(f"[{index}/{len(raw['images'])}] {key.name}" + (" recovered with crop" if recovered else ""), flush=True)
    for offset in range(0, len(tiles), 20):
        page = tiles[offset:offset+20]
        page += [np.zeros_like(tiles[0])] * (20-len(page))
        sheet = np.vstack([np.hstack(page[j:j+5]) for j in range(0,20,5)])
        write_image(out / "review" / f"sheet_{offset//20+1:02d}.jpg", sheet)
        full_page = full_tiles[offset:offset+20]
        full_page += [np.zeros_like(full_tiles[0])] * (20-len(full_page))
        write_image(out / "review" / f"full_{offset//20+1:02d}.jpg",
                    np.vstack([np.hstack(full_page[j:j+5]) for j in range(0,20,5)]))
    sessions = sorted({Path(row["key"]).parts[0] for row in records})
    if len(sessions) == 2:
        # Keep entire capture sessions separate; never randomly mix adjacent video frames.
        for split, session in zip(["train", "val"], sessions):
            paths = ["./images/" + row["key"] for row in records
                     if Path(row["key"]).parts[0] == session and not row["review_flags"]]
            (out / f"{split}.txt").write_text("\n".join(paths) + "\n")
        (out / "data.yaml").write_text(f"path: {json.dumps(str(out))}\ntrain: train.txt\nval: val.txt\nnames:\n  0: emergency_stop\n")
    (out / "manifest.json").write_text(json.dumps({
        "class_names": ["emergency_stop"], "target": "red button and yellow/black enclosure, excluding cable",
        "method": "BiRefNet on enclosure-centered crops, red-wire exclusion with button protection, 17px elliptical opening, yellow-seeded component selection, hole filling",
        "status": "draft automatic labels; inspect review sheets and full-resolution previews",
        "source_manifest": str((args.raw / "manifest.json").resolve()), "images": records,
        "split": "first session train, second session validation; flagged frames excluded" if len(sessions)==2 else None,
    }, indent=2) + "\n")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
