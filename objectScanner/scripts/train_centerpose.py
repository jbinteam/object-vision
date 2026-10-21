#!/usr/bin/env python3
"""Prepare objectScanner data and train the official NVlabs CenterPose model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CATEGORY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_category(value):
    if not CATEGORY_PATTERN.fullmatch(value):
        raise ValueError(
            "CenterPose category must use only letters, numbers, underscore, and hyphen")
    return value


def load_augmented_dataset(augmentation_root):
    augmentation_root = Path(augmentation_root).resolve()
    manifest_path = augmentation_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Augmentation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    category = validate_category(manifest["category"])
    if manifest.get("split") != "train":
        raise ValueError("CenterPose training input must have split=train")
    pair_dir = augmentation_root / "outf" / f"{category}_train"
    records = manifest.get("records", [])
    if not pair_dir.is_dir() or not records:
        raise ValueError("Augmented CenterPose pairs or source records are missing")

    seen_sources = set()
    for row in records:
        source_stem = Path(row["source_image"]).stem
        if not source_stem or source_stem in seen_sources:
            raise ValueError(f"Duplicate or empty source view: {source_stem!r}")
        seen_sources.add(source_stem)
        generated = row.get("generated", [])
        if not generated or Path(generated[0]["name"]).stem != source_stem:
            raise ValueError(f"Original pair must be first for source {source_stem}")
        for item in generated:
            name = Path(item["name"])
            if name.name != item["name"] or name.suffix.lower() != ".png":
                raise ValueError(f"Unsafe generated image name: {item['name']!r}")
            if not (pair_dir / name.name).is_file():
                raise ValueError(f"Missing augmented image: {name.name}")
            if not (pair_dir / name.with_suffix(".json")).is_file():
                raise ValueError(f"Missing CenterPose label: {name.with_suffix('.json').name}")
    return manifest_path, manifest, pair_dir, records


def split_sources(records, val_fraction, seed):
    if not 0 < val_fraction < 0.5:
        raise ValueError("Validation fraction must be greater than 0 and less than 0.5")
    if len(records) < 4:
        raise ValueError("At least four source views are required for a train/validation split")
    val_count = max(1, min(len(records) - 1, int(round(len(records) * val_fraction))))
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return {Path(records[index]["source_image"]).stem for index in indices[:val_count]}


def symlink_pair(source_dir, destination_dir, image_name):
    image_name = Path(image_name)
    label_name = image_name.with_suffix(".json")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in (image_name, label_name):
        source = (source_dir / name).resolve()
        if not source.is_file():
            raise ValueError(f"Missing CenterPose pair member: {source}")
        destination = destination_dir / name
        destination.symlink_to(os.path.relpath(source, destination.parent))


def validate_prepared_split(training_root, category, expected=None):
    training_root = Path(training_root).resolve()
    result = {}
    for split in ("train", "val"):
        directory = training_root / "outf" / f"{category}_{split}"
        images = sorted(directory.glob("*.png")) if directory.is_dir() else []
        labels = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if not images or len(images) != len(labels):
            raise ValueError(f"Invalid {split} pair count in {directory}")
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ValueError(f"Unpaired PNG/JSON files in {directory}")
        result[split] = len(images)
    if expected and result != expected:
        raise ValueError(f"Prepared pair counts are {result}, expected {expected}")
    return result


def prepare_training_split(augmentation_root, training_root, val_fraction=0.125,
                           seed=317, rebuild=False):
    manifest_path, manifest, pair_dir, records = load_augmented_dataset(augmentation_root)
    category = manifest["category"]
    training_root = Path(training_root).resolve()
    val_sources = split_sources(records, val_fraction, seed)
    expected_train = sum(
        len(row["generated"]) for row in records
        if Path(row["source_image"]).stem not in val_sources)
    expected = {"train": expected_train, "val": len(val_sources)}
    signature = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "category": category,
        "seed": seed,
        "validation_fraction": val_fraction,
        "validation_policy": "source-grouped; original image only",
        "validation_source_stems": sorted(val_sources),
        "source_views": len(records),
        "pairs": expected,
    }

    split_manifest_path = training_root / "split.json"
    if training_root.exists():
        if split_manifest_path.is_file() and json.loads(split_manifest_path.read_text()) == signature:
            validate_prepared_split(training_root, category, expected)
            return signature
        if not rebuild:
            raise ValueError(
                f"Prepared split already exists with different settings: {training_root}; "
                "pass --rebuild-split to replace it")

    training_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{training_root.name}-building-", dir=training_root.parent))
    try:
        train_dir = stage / "outf" / f"{category}_train"
        val_dir = stage / "outf" / f"{category}_val"
        for row in records:
            source_stem = Path(row["source_image"]).stem
            if source_stem in val_sources:
                # Validate on the unaugmented capture and keep all of its synthetic
                # variants out of training to prevent source-view leakage.
                symlink_pair(pair_dir, val_dir, row["generated"][0]["name"])
            else:
                for item in row["generated"]:
                    symlink_pair(pair_dir, train_dir, item["name"])
        validate_prepared_split(stage, category, expected)
        write_json(stage / "split.json", signature)
        if training_root.exists():
            backup = training_root.with_name(
                f".{training_root.name}-replaced-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            training_root.rename(backup)
            try:
                stage.rename(training_root)
            except BaseException:
                backup.rename(training_root)
                raise
            shutil.rmtree(backup)
        else:
            stage.rename(training_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return signature


def validate_centerpose_checkout(centerpose_root):
    centerpose_root = Path(centerpose_root).resolve()
    required = [
        centerpose_root / "src/main_CenterPose.py",
        centerpose_root / "src/lib/opts.py",
        centerpose_root / "src/lib/datasets/dataset_combined.py",
        centerpose_root / "src/lib/models/networks/DCNv2/dcn_v2.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "Not an NVlabs/CenterPose checkout; missing: " + ", ".join(missing))
    return centerpose_root


def run_training(args, split):
    centerpose_root = validate_centerpose_checkout(args.centerpose_root)
    src = centerpose_root / "src"
    sys.path.insert(0, str(src))
    if args.gpus != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    try:
        import torch
        from lib.opts import opts
        from main_CenterPose import main
    except ModuleNotFoundError as error:
        if error.name in {"_ext", "DCNv2._ext"}:
            raise ValueError(
                "CenterPose still uses its legacy DCNv2 extension. Apply "
                "patches/centerpose-modern-pytorch.patch to the CenterPose checkout") from error
        raise ValueError(
            f"CenterPose environment is missing Python dependency {error.name!r}; "
            "run this launcher with the CenterPose environment's Python") from error

    if not torch.cuda.is_available() and args.gpus != "-1":
        raise ValueError("CUDA is unavailable in this Python environment; use a CUDA build of PyTorch")
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    option_builder = opts()
    opt = option_builder.parser.parse_args([])
    opt.c = split["category"]
    opt.arch = args.arch
    opt.obj_scale = True
    opt.obj_scale_weight = 1
    opt.mug = False
    opt.exp_id = args.run_name or f"objectron_{opt.c}_{opt.arch}"
    opt.num_epochs = args.epochs
    opt.num_iters = args.num_iters
    opt.val_intervals = args.val_interval
    opt.lr_step = args.lr_step
    opt.batch_size = args.batch_size
    opt.lr = args.learning_rate if args.learning_rate is not None else 6e-5 * args.batch_size / 16
    opt.gpus = args.gpus
    opt.num_workers = args.workers
    opt.print_iter = args.print_interval
    opt.debug = 0
    opt.save_all = args.save_all
    opt.load_model = str(args.load_model.resolve()) if args.load_model else ""
    opt.resume = args.resume
    opt = option_builder.parse(opt)

    opt.data_dir = str(args.training_root.resolve())
    opt.exp_dir = str(args.output_dir.resolve())
    if args.run_name:
        run_name = args.run_name
    elif args.resume and args.load_model:
        run_name = args.load_model.resolve().parent.name
    else:
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        run_name = f"objectron_{opt.c}_{opt.arch}_{stamp}"
    opt.save_dir = str(args.output_dir.resolve() / run_name)
    opt.debug_dir = str(Path(opt.save_dir) / "debug")
    if args.resume and not args.load_model:
        raise ValueError("--resume requires --load-model so the optimizer state can be restored")
    if Path(opt.save_dir).exists() and not args.resume:
        raise ValueError(f"Training output already exists: {opt.save_dir}")

    launch = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "centerpose_root": str(centerpose_root),
        "training_data": str(args.training_root.resolve()),
        "split": split,
        "architecture": opt.arch,
        "epochs": opt.num_epochs,
        "iterations_per_epoch": opt.num_iters,
        "batch_size": opt.batch_size,
        "learning_rate": opt.lr,
        "lr_steps": opt.lr_step,
        "gpu_ids": args.gpus,
        "output": opt.save_dir,
    }
    print(json.dumps(launch, indent=2), flush=True)
    if args.dry_run:
        return Path(opt.save_dir)
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(opt.save_dir) / "objectscanner_launch.json", launch)
    main(opt)
    return Path(opt.save_dir)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "datasets/my_object_contrast/centerpose_augmented",
        help="Background-augmented dataset root containing manifest.json")
    parser.add_argument(
        "--training-root", type=Path,
        help="Prepared lightweight train/val split (symlinks; default: dataset sibling)")
    parser.add_argument(
        "--centerpose-root", type=Path,
        default=ROOT.parent / "CenterPose",
        help="Official NVlabs/CenterPose checkout")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "training/centerpose",
        help="Directory for checkpoints and TensorBoard logs")
    parser.add_argument("--run-name", help="Stable output folder name; timestamped by default")
    parser.add_argument("--val-fraction", type=float, default=0.125,
                        help="Held-out source-view fraction (default: 0.125)")
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare/validate the split without importing CenterPose")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate CenterPose and print the resolved training configuration")
    parser.add_argument("--arch", default="dlav1_34", choices=("dlav1_34",),
                        help="Benchmark CenterPose DLA-34 + convGRU architecture")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--num-iters", type=int, default=-1,
                        help="Iterations per epoch; -1 uses the complete training split")
    parser.add_argument("--lr-step", default="90,120")
    parser.add_argument("--learning-rate", type=float,
                        help="Default scales benchmark 6e-5 at batch 16 linearly")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0
            or args.num_iters == 0 or args.num_iters < -1):
        parser.error(
            "epochs and batch size must be positive; workers cannot be negative; "
            "--num-iters must be -1 or positive")
    if args.val_interval < 1:
        parser.error("--val-interval must be positive so checkpoints are saved")
    try:
        [int(value) for value in args.lr_step.split(",")]
    except ValueError:
        parser.error("--lr-step must be comma-separated integer epochs")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.training_root is None:
        args.training_root = args.dataset.resolve().parent / "centerpose_training"
    split = prepare_training_split(
        args.dataset, args.training_root, args.val_fraction, args.seed,
        args.rebuild_split)
    print(
        f"Prepared CenterPose split: {split['pairs']['train']} train, "
        f"{split['pairs']['val']} validation ({args.training_root.resolve()})",
        flush=True,
    )
    if args.prepare_only:
        return 0
    run_training(args, split)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"CenterPose training error: {error}")
