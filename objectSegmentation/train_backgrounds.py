"""Preview or train YOLO segmentation with a newly sampled background on each read."""
import argparse
from functools import partial
import json
import hashlib
import math
from pathlib import Path
import random

import cv2
import numpy as np

from background_augmentation import ROOT, BackgroundSegmentationTrainer, make_training_dataset
from ultralytics import YOLO, settings
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    latest = ROOT / "output/l515_dataset/latest/data.yaml"
    parser.add_argument("--data", type=Path, default=latest if latest.is_file() else ROOT / "output/l515_emergency_stop/data.yaml")
    parser.add_argument("--backgrounds", type=Path, default=ROOT / "backgrounds/picsum")
    parser.add_argument("--background-probability", type=float, default=0.85)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0", help="Single GPU index or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="emergency_stop_backgrounds")
    parser.add_argument("--freeze", type=int, default=0, metavar="N", help="Freeze the first N model modules; YOLO11 backbone = 11; 0 trains all")
    parser.add_argument("--optimizer", choices=["auto", "AdamW", "SGD"], default="auto")
    parser.add_argument("--lr0", type=float, help="Initial learning rate; requires an explicit optimizer")
    parser.add_argument("--preview", type=int, default=0, metavar="N", help="Save N augmented samples and exit without training")
    parser.add_argument("--preview-output", type=Path, default=ROOT / "output/background_preview")
    args = parser.parse_args(argv)
    if not 0 <= args.background_probability <= 1:
        parser.error("--background-probability must be between 0 and 1")
    if min(args.epochs, args.batch, args.imgsz) <= 0 or args.workers < 0 or args.preview < 0:
        parser.error("epochs, batch, and imgsz must be positive; workers and preview nonnegative")
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("Use one GPU index or cpu; this wrapper does not configure distributed training")
    if args.freeze < 0:
        parser.error("--freeze must be nonnegative")
    if args.lr0 is not None:
        if not math.isfinite(args.lr0) or args.lr0 <= 0:
            parser.error("--lr0 must be finite and positive")
        if args.optimizer == "auto":
            parser.error("--lr0 requires --optimizer AdamW or SGD; auto would ignore the requested learning rate")
    return args


class FreezeAudit:
    """Verify requested modules stay unchanged, including BatchNorm running state."""
    def __init__(self, count):
        self.prefixes = tuple(f"model.{i}." for i in range(count))
        self.initial = None

    def digest(self, model):
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            if name.startswith(self.prefixes):
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def start(self, trainer):
        frozen = [(n, p) for n, p in trainer.model.named_parameters() if n.startswith(self.prefixes)]
        if any(p.requires_grad for _, p in frozen):
            raise RuntimeError("Requested frozen modules still have trainable parameters")
        self.initial = self.digest(trainer.model)
        self.record = {"source_checkpoint": str(trainer.args.model),
                       "freeze": trainer.args.freeze, "optimizer": trainer.args.optimizer,
                       "lr0": trainer.args.lr0, "warmup_bias_lr": trainer.args.warmup_bias_lr,
                       "frozen_parameters": sum(p.numel() for _, p in frozen),
                       "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                       "initial_frozen_state_sha256": self.initial,
                       "frozen_state_unchanged": None}
        self.path = trainer.save_dir / "freeze_audit.json"
        self.path.write_text(json.dumps(self.record, indent=2) + "\n")

    def finish(self, trainer):
        final = self.digest(trainer.model)
        self.record.update(final_frozen_state_sha256=final, frozen_state_unchanged=final == self.initial)
        self.path.write_text(json.dumps(self.record, indent=2) + "\n")
        if final != self.initial:
            raise RuntimeError("Frozen weights or BatchNorm state changed during training")


def main():
    args = parse_args()
    # Settings live under this project's .codex folder, not the user's global config.
    settings.update({key: False for key in ("sync", "hub", "wandb", "comet", "clearml", "mlflow",
                                            "neptune", "raytune", "tensorboard") if key in settings})
    options = dict(data=str(args.data.resolve()), task="segment", epochs=args.epochs,
                   batch=args.batch, imgsz=args.imgsz, workers=args.workers, device=args.device,
                   seed=args.seed, project=str(ROOT / "runs/segment"), name=args.name,
                   cache=False, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                   degrees=15.0, translate=0.10, scale=0.40, fliplr=0.5, flipud=0.0,
                   hsv_h=0.015, hsv_s=0.35, hsv_v=0.25, patience=25, plots=True,
                   freeze=args.freeze, optimizer=args.optimizer, resume=False)
    if args.lr0 is not None:
        # Avoid the upstream 0.1 bias warmup rate overwhelming a small fine-tuning LR.
        options.update(lr0=args.lr0, warmup_bias_lr=0.0)
    if args.preview:
        random.seed(args.seed)
        np.random.seed(args.seed)
        data = check_det_dataset(options["data"])
        dataset = make_training_dataset(
            data["train"], data,
            get_cfg(overrides=options), args.batch, args.backgrounds, args.background_probability,
        )
        out = args.preview_output.resolve()
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"Preview already exists: {out}; choose a new --preview-output directory")
        out.mkdir(parents=True, exist_ok=True)
        tiles = []
        for index in range(args.preview):
            # Same image repeatedly for the first four samples demonstrates on-the-fly variation.
            sample = dataset[0 if index < 4 else (index-3) % len(dataset)]
            image = sample["img"].permute(1, 2, 0).numpy()[:, :, ::-1].copy()
            cv2.imwrite(str(out / f"sample_{index+1:03d}.jpg"), image)
            mask = sample["masks"].numpy()
            mask = np.any(mask > 0, axis=0).astype(np.uint8) if len(mask) else np.zeros((args.imgsz, args.imgsz), np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = image.copy()
            overlay[mask > 0] = (overlay[mask > 0]*0.65 + np.array([0, 255, 0])*0.35).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
            cv2.imwrite(str(out / f"sample_{index+1:03d}_labels.jpg"), overlay)
            tile = cv2.resize(overlay, (320, 320))
            cv2.putText(tile, str(index+1), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            tiles.append(tile)
        for offset in range(0, len(tiles), 12):
            page = tiles[offset:offset+12]
            page += [np.zeros_like(tiles[0])] * (12-len(page))
            cv2.imwrite(str(out / f"contact_{offset//12+1:02d}.jpg"), np.vstack([np.hstack(page[j:j+4]) for j in range(0,12,4)]))
        (out / "settings.json").write_text(json.dumps({**options, "background_probability": args.background_probability,
                                                      "backgrounds": str(args.backgrounds.resolve()),
                                                      "first_four_samples": "same source image, independently augmented"}, indent=2)+"\n")
        print(f"Preview: {out}")
        return
    trainer = partial(BackgroundSegmentationTrainer, background_dir=args.backgrounds,
                      background_probability=args.background_probability)
    model = YOLO(args.model)
    if model.task != "segment":
        raise ValueError("Choose a segmentation checkpoint")
    if args.freeze >= len(model.model.model):
        raise ValueError("--freeze would freeze the whole model; leave trainable modules")
    if args.freeze:
        audit = FreezeAudit(args.freeze)
        model.add_callback("on_train_start", audit.start)
        model.add_callback("on_train_end", audit.finish)
    model.train(trainer=trainer, **options)


if __name__ == "__main__":
    main()
