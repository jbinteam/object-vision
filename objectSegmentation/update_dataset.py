"""Add new L515 PNG captures to versioned emergency-stop datasets; optionally train."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

import cv2
import numpy as np

from prepare_emergency_stop import ROOT, CropPredictor, prepare_image, write_image

PRODUCTS = ("images", "labels", "masks", "cutouts", "previews", "crop_masks")


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def identity(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pixel_sha256": hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()}


def product_path(root, product, key):
    key = Path(key)
    if key.is_absolute() or ".." in key.parts:
        raise ValueError(f"Unsafe dataset key: {key}")
    return (root / product / key).with_suffix(".txt") if product == "labels" else root / product / key


def validate_annotation(root, row):
    """Check real saved products, including polygon-to-mask agreement."""
    key = row["key"]
    image = cv2.imread(str(product_path(root, "images", key)))
    mask = cv2.imread(str(product_path(root, "masks", key)), 0)
    cutout = cv2.imread(str(product_path(root, "cutouts", key)), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None or image.shape[:2] != mask.shape:
        raise ValueError(f"Missing or wrong-sized image/mask: {key}")
    if not np.isin(mask, [0, 255]).all() or not mask.any():
        raise ValueError(f"Nonbinary or empty foreground mask: {key}")
    if cutout is None or cutout.shape != (*mask.shape, 4) or not np.array_equal(cutout[:, :, 3], mask):
        raise ValueError(f"Cutout alpha differs from mask: {key}")
    lines = product_path(root, "labels", key).read_text().splitlines()
    if len(lines) != 1:
        raise ValueError(f"Expected one target polygon: {key}")
    values = np.array([float(v) for v in lines[0].split()])
    if len(values) < 7 or len(values) % 2 != 1 or values[0] != 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid YOLO label: {key}")
    xy = values[1:].reshape(-1, 2)
    if (xy < 0).any() or (xy > 1).any():
        raise ValueError(f"Out-of-range polygon: {key}")
    polygon_mask = np.zeros_like(mask)
    cv2.fillPoly(polygon_mask, [np.rint(xy * [mask.shape[1], mask.shape[0]]).astype(np.int32)], 255)
    intersection = np.count_nonzero((mask > 0) & (polygon_mask > 0))
    union = np.count_nonzero((mask > 0) | (polygon_mask > 0))
    iou = intersection / union
    if iou < .98:
        raise ValueError(f"Polygon/mask disagreement: {key} (IoU {iou:.4f})")
    if hashlib.sha256(product_path(root, "images", key).read_bytes()).hexdigest() != row["source_sha256"]:
        raise ValueError(f"Image changed after labeling: {key}")
    return iou


def import_base(root):
    manifest = json.loads((root / "manifest.json").read_text())
    splits = {}
    sessions = {}
    for split in ("train", "val"):
        for line in (root / f"{split}.txt").read_text().splitlines():
            if not line.strip():
                continue
            path = Path(line)
            key = str(path.resolve().relative_to(root / "images")) if path.is_absolute() else str(path.relative_to("images"))
            if key in splits:
                raise ValueError(f"Base train/val overlap: {key}")
            splits[key] = split
            session = Path(key).parts[0]
            if session in sessions and sessions[session] != split:
                raise ValueError(f"Base session split between train and val: {session}")
            sessions[session] = split
    rows = []
    for original in manifest["images"]:
        row = {**original, **identity(root / "images" / original["key"])}
        row.update(status="labeled", split=splits.get(row["key"], "excluded"), origin="base")
        rows.append(row)
    return rows, sessions


def choose_split(session, assignments, validation_sessions):
    requested = "val" if session in validation_sessions else "train"
    if session in assignments:
        if session in validation_sessions and assignments[session] != "val":
            raise ValueError(f"Session {session} already belongs to training; capture a fresh validation session")
        return assignments[session]
    assignments[session] = requested
    return requested


def review_sheets(root, rows):
    (root / "review").mkdir(parents=True, exist_ok=True)
    labeled = [r for r in rows if r["status"] == "labeled"]
    for offset in range(0, len(labeled), 20):
        crops, full = [], []
        for row in labeled[offset:offset + 20]:
            preview = cv2.imread(str(root / "previews" / row["key"]))
            x, y, width, height = row["bbox_xywh"]
            crop = preview[max(0, y-30):y+height+30, max(0, x-30):x+width+30]
            scale = min(360 / crop.shape[1], 250 / crop.shape[0])
            crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
            canvas = np.full((290, 380, 3), 32, np.uint8)
            canvas[35:35+crop.shape[0], :crop.shape[1]] = crop
            title = f"{row['index']:03d} {row['split']} {Path(row['key']).name[:12]}"
            cv2.putText(canvas, title, (5, 22), 0, .45, (255, 255, 255), 1)
            crops.append(canvas)
            tile = cv2.resize(preview, (480, 270))
            cv2.putText(tile, title, (8, 24), 0, .55, (0, 0, 255), 2)
            full.append(tile)
        for prefix, tiles in (("crops", crops), ("full", full)):
            tiles += [np.zeros_like(tiles[0])] * (20-len(tiles))
            write_image(root / "review" / f"{prefix}_{offset//20+1:02d}.jpg",
                        np.vstack([np.hstack(tiles[i:i+5]) for i in range(0, 20, 5)]))
    (root / "review" / "index.json").write_text(json.dumps(rows, indent=2) + "\n")


def update(args, predictor_factory=CropPredictor):
    source, store, base = args.input.resolve(), args.dataset.resolve(), args.base.resolve()
    latest = store / "latest"
    previous = latest.resolve() if latest.exists() else base
    if previous == base:
        rows, assignments = import_base(base)
    else:
        old = json.loads((previous / "manifest.json").read_text())
        rows, assignments = old["images"], old["session_assignments"].copy()
        if old["input"] != str(source):
            raise ValueError("Use the same --input root on every update")
    by_key = {r["key"]: r for r in rows}
    seen_pixels = {r["pixel_sha256"]: r["key"] for r in rows if r["status"] == "labeled"}
    files = sorted(source.glob("*/*.png"))
    if not files:
        raise ValueError(f"No session PNG captures found under {source}")
    unknown_val = set(args.val_session) - {p.parent.name for p in files} - set(assignments)
    if unknown_val:
        raise ValueError(f"Unknown validation sessions: {sorted(unknown_val)}")
    for session in args.val_session:
        choose_split(session, assignments, set(args.val_session))
    pending = []
    for path in files:
        key = str(path.relative_to(source))
        fingerprint = identity(path)
        old = by_key.get(key)
        if old:
            if old["source_sha256"] != fingerprint["source_sha256"]:
                raise ValueError(f"Previously imported source changed: {path}. Put revised captures in a new session.")
            if old["status"] != "error":
                continue
        pending.append((path, key, fingerprint))
    print(f"Existing records: {len(rows)} | new/retry images: {len(pending)}", flush=True)
    if args.dry_run:
        for path, key, _ in pending:
            split = choose_split(path.parent.name, assignments, set(args.val_session))
            print(f"  {split}: {key}")
        return previous
    if not pending and latest.exists():
        print(f"No new captures. Reusing {previous}", flush=True)
        return previous
    origins = {r["key"]: previous for r in rows}
    predictor = None
    # Changing labeling code invalidates interrupted-work caches, not published labels.
    algorithm = hashlib.sha256((ROOT / "prepare_emergency_stop.py").read_bytes()).hexdigest()[:16]
    new_keys = []
    for index, (path, key, fingerprint) in enumerate(pending, 1):
        session = path.parent.name
        split = choose_split(session, assignments, set(args.val_session))
        row = {"source": str(path), "key": key, **fingerprint, "origin": "incremental"}
        duplicate = seen_pixels.get(fingerprint["pixel_sha256"])
        if duplicate and duplicate != key:
            row.update(status="duplicate", split="excluded", duplicate_of=duplicate,
                       review_flags=["exact decoded-image duplicate; excluded to prevent leakage"])
        else:
            token = hashlib.sha256((algorithm + key + fingerprint["source_sha256"]).encode()).hexdigest()
            cache = store / "cache" / token
            record_path = cache / "record.json"
            try:
                if record_path.is_file():
                    cached = json.loads(record_path.read_text())
                    validate_annotation(cache, cached)
                else:
                    if predictor is None:
                        predictor = predictor_factory()
                    cached = {**prepare_image(path, key, cache, predictor), **fingerprint}
                    validate_annotation(cache, cached)
                    atomic_json(record_path, cached)
                row.update(cached)
                row.update(status="labeled", split="excluded" if row["review_flags"] else split)
                seen_pixels[fingerprint["pixel_sha256"]] = key
                origins[key] = cache
            except ValueError as error:
                row.update(status="error", split="excluded", review_flags=[str(error)])
        by_key[key] = row
        new_keys.append(key)
        print(f"[{index}/{len(pending)}] {path.name}: {row['status']} / {row['split']}" +
              (f" ({'; '.join(row['review_flags'])})" if row["review_flags"] else ""), flush=True)
    records = list(by_key.values())
    versions = store / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = versions / revision
    stage = Path(tempfile.mkdtemp(prefix=".building-", dir=versions))
    try:
        ious = []
        for index, row in enumerate(records, 1):
            row["index"] = index
            if row["status"] != "labeled":
                continue
            origin = origins[row["key"]]
            for product in PRODUCTS:
                src = product_path(origin, product, row["key"])
                dst = product_path(stage, product, row["key"])
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            ious.append(validate_annotation(stage, row))
        counts = dict(Counter(row["split"] for row in records))
        for split in ("train", "val"):
            selected = [r for r in records if r["split"] == split]
            if not selected:
                raise ValueError(f"The dataset has no {split} images")
            (stage / f"{split}.txt").write_text("".join(f"./images/{r['key']}\n" for r in selected))
        train_sessions = {Path(r["key"]).parts[0] for r in records if r["split"] == "train"}
        val_sessions = {Path(r["key"]).parts[0] for r in records if r["split"] == "val"}
        if train_sessions & val_sessions:
            raise ValueError("Capture session overlaps training and validation")
        active = [r["pixel_sha256"] for r in records if r["split"] in ("train", "val")]
        if len(active) != len(set(active)):
            raise ValueError("Duplicate images in the active dataset")
        updated_sessions = {Path(key).parts[0] for key in new_keys}
        review_sheets(stage, [r for r in records if Path(r["key"]).parts[0] in updated_sessions])
        manifest = {"schema_version": 1, "input": str(source), "parent": str(previous),
                    "revision": revision, "class_names": ["emergency_stop"],
                    "target": "red button and yellow/black enclosure, excluding cable",
                    "status": "draft automatic labels; inspect review sheets and previews",
                    "session_assignments": assignments, "added_or_retried": new_keys,
                    "images": records}
        atomic_json(stage / "manifest.json", manifest)
        report = {"records": len(records), "labeled": len(ious), "split_counts": counts,
                  "added_or_retried": len(pending), "polygon_mask_iou_min": min(ious),
                  "polygon_mask_iou_mean": float(np.mean(ious)),
                  "review_required": [{"key": r["key"], "flags": r.get("review_flags", [])}
                                      for r in records if r["split"] == "excluded"],
                  "checks": "source checksum, binary masks, cutout alpha, normalized polygons, polygon-mask IoU >= .98, session separation, exact pixel deduplication"}
        atomic_json(stage / "validation.json", report)
        (stage / "data.yaml").write_text(f"path: {json.dumps(str(destination))}\ntrain: train.txt\nval: val.txt\nnames:\n  0: emergency_stop\n")
        stage.rename(destination)
        temporary_link = store / ".latest.tmp"
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(destination.relative_to(store), target_is_directory=True)
        temporary_link.replace(latest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"Dataset: {destination}\nSplits: {counts}\nReview: {destination / 'review'}", flush=True)
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "input/l515")
    parser.add_argument("--base", type=Path, default=ROOT / "output/l515_emergency_stop")
    parser.add_argument("--dataset", type=Path, default=ROOT / "output/l515_dataset")
    parser.add_argument("--val-session", action="append", default=[], help="Assign an entirely NEW capture session to validation")
    parser.add_argument("--dry-run", action="store_true", help="List pending captures without labeling or publishing")
    parser.add_argument("--train", action="store_true", help="After updating, start a new fine-tuning run with fresh optimizer state")
    parser.add_argument("--model", type=Path, default=ROOT / "runs/segment/emergency_stop_backgrounds/weights/best.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--freeze", type=int, default=11, help="Freeze first N modules; default 11 = YOLO11 backbone; 0 trains all")
    parser.add_argument("--optimizer", choices=["AdamW", "SGD"], default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0001, help="Fine-tuning learning rate")
    args = parser.parse_args(argv)
    if args.train and args.dry_run:
        parser.error("--train and --dry-run cannot be combined")
    if min(args.epochs, args.batch) <= 0:
        parser.error("--epochs and --batch must be positive")
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("--device must be cpu or a GPU index")
    if args.freeze < 0 or not math.isfinite(args.lr0) or args.lr0 <= 0:
        parser.error("--freeze must be nonnegative and --lr0 finite and positive")
    if args.train and not args.model.is_file():
        parser.error(f"Checkpoint missing: {args.model}")
    return args


def main():
    args = parse_args()
    args.dataset.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock prevents simultaneous updates from publishing competing manifests.
    with (args.dataset / ".update.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another dataset update is running")
        dataset = update(args)
    command = [sys.executable, "-E", str(ROOT / "train_backgrounds.py"), "--data", str(dataset / "data.yaml"),
               "--model", str(args.model.resolve()), "--epochs", str(args.epochs), "--batch", str(args.batch),
               "--device", args.device, "--name", f"emergency_stop_{dataset.name}",
               "--freeze", str(args.freeze), "--optimizer", args.optimizer, "--lr0", str(args.lr0)]
    if not args.dry_run:
        print("Train command:\n" + shlex.join(command), flush=True)
    if args.train:
        # Replacing this process releases BiRefNet GPU memory before YOLO loads.
        os.execv(sys.executable, command)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Dataset update error: {error}")
