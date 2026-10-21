#!/usr/bin/env python3
"""Train synchronized Ultralytics segmentation and 8-keypoint pose models."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SEGMENT_MODEL = (
    ROOT.parent / "objectSegmentation/runs/segment/"
    "emergency_stop_frozen_backbone/weights/best.pt")
DEFAULT_BACKGROUNDS = ROOT.parent / "objectSegmentation/backgrounds/picsum"


def paired_counts(dataset):
    dataset = Path(dataset).resolve()
    result = {}
    for task in ("segmentation", "pose"):
        yaml = dataset / task / "data.yaml"
        if not yaml.is_file():
            raise ValueError(f"Missing {task} dataset YAML: {yaml}")
        result[task] = {}
        for split in ("train", "val"):
            images = list((dataset / task / "images" / split).glob("*"))
            labels = list((dataset / task / "labels" / split).glob("*.txt"))
            masks = list((dataset / task / "masks" / split).glob("*.png"))
            if not images or len(images) != len(labels):
                raise ValueError(
                    f"{task}/{split} has {len(images)} images and {len(labels)} labels")
            if {path.stem for path in images} != {path.stem for path in labels}:
                raise ValueError(f"Unpaired {task}/{split} sample stems")
            if len(masks) != len(images) or {path.stem for path in masks} != {
                    path.stem for path in images}:
                raise ValueError(f"Unpaired {task}/{split} foreground masks")
            result[task][split] = len(images)
    if result["segmentation"] != result["pose"]:
        raise ValueError("Segmentation and pose split counts differ")
    return result


def train_task(args, task, model_path):
    from ultralytics import YOLO
    from yolo_background_augmentation import (
        BackgroundPoseTrainer, BackgroundSegmentationTrainer)

    data = args.dataset.resolve() / task / "data.yaml"
    model = YOLO(str(model_path), task="segment" if task == "segmentation" else "pose")
    name = f"{args.name}_{'seg' if task == 'segmentation' else 'pose'}"
    options = dict(
        data=str(data),
        project=str(args.project.resolve()),
        name=name,
        exist_ok=args.exist_ok,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=args.lr0,
        freeze=args.freeze,
        seed=args.seed,
        deterministic=True,
        fliplr=0.0,
        close_mosaic=min(10, max(0, args.epochs // 5)),
        cache=args.cache,
        plots=True,
        verbose=True,
    )
    if not args.no_background_augmentation:
        trainer_class = (BackgroundSegmentationTrainer
                         if task == "segmentation" else BackgroundPoseTrainer)
        options["trainer"] = partial(
            trainer_class, background_dir=args.backgrounds,
            background_probability=args.background_probability,
            epoch_repeats=args.epoch_repeats)
    return model.train(**options)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "datasets/my_object_rgbd/yolo_seg_pose_onthefly")
    parser.add_argument("--project", type=Path, default=ROOT / "training/yolo")
    parser.add_argument("--name", default="my_object_rgbd")
    parser.add_argument("--task", choices=("both", "segmentation", "pose"), default="both")
    parser.add_argument("--segment-model", type=Path, default=DEFAULT_SEGMENT_MODEL)
    parser.add_argument("--pose-model", default="yolo11n-pose.pt",
                        help="Local checkpoint or Ultralytics model name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--freeze", type=int, default=11,
                        help="Freeze modules 0 through N-1 (default: complete YOLO11 backbone)")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--backgrounds", type=Path, default=DEFAULT_BACKGROUNDS,
                        help="Background image library sampled independently on every training read")
    parser.add_argument("--background-probability", type=float, default=0.85)
    parser.add_argument(
        "--epoch-repeats", type=int, default=12,
        help=("Training draws per source image per epoch (default: 12); "
              "each draw independently samples an on-the-fly background"))
    parser.add_argument("--no-background-augmentation", action="store_true",
                        help="Keep original training backgrounds (validation is always unchanged)")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch, args.imgsz,
           args.epoch_repeats) < 1 or args.workers < 0:
        parser.error("Epochs, patience, batch, and image size must be positive")
    if args.freeze < 0 or args.lr0 <= 0:
        parser.error("Freeze cannot be negative and lr0 must be positive")
    if not 0 <= args.background_probability <= 1:
        parser.error("Background probability must be between 0 and 1")
    args.cache = False if args.cache == "false" else args.cache
    return args


def main(argv=None):
    args = parse_args(argv)
    counts = paired_counts(args.dataset)
    plan = {
        "dataset": str(args.dataset.resolve()),
        "counts": counts,
        "task": args.task,
        "segment_model": str(args.segment_model),
        "pose_model": str(args.pose_model),
        "project": str(args.project.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "freeze": args.freeze,
        "fliplr": 0.0,
        "backgrounds": str(args.backgrounds.resolve()),
        "background_probability": (0.0 if args.no_background_augmentation
                                   else args.background_probability),
        "epoch_repeats": (1 if args.no_background_augmentation
                          else args.epoch_repeats),
        "background_mode": "on-the-fly training only",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    if args.task in ("both", "segmentation"):
        if not args.segment_model.is_file():
            raise ValueError(f"Segmentation checkpoint does not exist: {args.segment_model}")
        train_task(args, "segmentation", args.segment_model)
    if args.task in ("both", "pose"):
        train_task(args, "pose", args.pose_model)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"YOLO paired training error: {error}")
