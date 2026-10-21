"""On-the-fly, training-only background replacement for YOLO segmentation."""
from functools import lru_cache
import json
import os
from pathlib import Path
import random

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
(ROOT / ".codex/ultralytics").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".codex/ultralytics")

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.torch_utils import unwrap_model


class BackgroundReplacer:
    def __init__(self, directory):
        directory = Path(directory).resolve()
        manifest = directory / "sources.json"
        if manifest.is_file():
            self.paths = [directory / row["file"] for row in json.loads(manifest.read_text())["images"]]
        else:
            self.paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.paths or not all(p.is_file() for p in self.paths):
            raise ValueError(f"Backgrounds missing in {directory}; run download_backgrounds.py")

    @staticmethod
    @lru_cache(maxsize=16)
    def read(path):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable background: {path}")
        return image

    def composite(self, image, mask, rng=random):
        if mask.shape != image.shape[:2]:
            raise ValueError("Image and foreground mask dimensions must match")
        height, width = image.shape[:2]
        background = self.read(rng.choice(self.paths))
        scale = max(width / background.shape[1], height / background.shape[0]) * rng.uniform(1.0, 1.3)
        resized = cv2.resize(background, (max(width, int(np.ceil(background.shape[1]*scale))),
                                         max(height, int(np.ceil(background.shape[0]*scale)))))
        x = rng.randint(0, resized.shape[1] - width)
        y = rng.randint(0, resized.shape[0] - height)
        canvas = resized[y:y+height, x:x+width].copy()
        if rng.random() < 0.5:
            canvas = canvas[:, ::-1].copy()
        # Keep foreground pixels and their geometry unchanged. Labels remain exact.
        canvas[mask > 0] = image[mask > 0]
        return canvas


class BackgroundYOLODataset(YOLODataset):
    def __init__(self, *args, dataset_root, background_dir, background_probability=0.85, **kwargs):
        self.dataset_root = Path(dataset_root).resolve()
        self.background_probability = background_probability
        self.background_replacer = BackgroundReplacer(background_dir)
        super().__init__(*args, **kwargs)
        self.mask_paths = {}
        for image in self.im_files:
            path = self.dataset_root / "masks" / Path(image).resolve().relative_to(self.dataset_root / "images")
            if not path.is_file():
                raise ValueError(f"Foreground mask missing: {path}")
            self.mask_paths[image] = path

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        if self.augment and random.random() < self.background_probability:
            mask = cv2.imread(str(self.mask_paths[label["im_file"]]), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != label["ori_shape"]:
                raise ValueError(f"Invalid foreground mask for {label['im_file']}")
            height, width = label["img"].shape[:2]
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            label["img"] = self.background_replacer.composite(label["img"], mask)
        return label


def make_training_dataset(img_path, data, hyp, batch, background_dir, probability, stride=32):
    return BackgroundYOLODataset(
        img_path=img_path, imgsz=hyp.imgsz, batch_size=batch, augment=True,
        hyp=hyp, rect=hyp.rect, cache=hyp.cache or None, single_cls=hyp.single_cls,
        stride=stride, pad=0.0, prefix="train: ", task="segment", classes=hyp.classes,
        data=data, fraction=hyp.fraction, dataset_root=data["path"],
        background_dir=background_dir, background_probability=probability,
    )


class BackgroundSegmentationTrainer(SegmentationTrainer):
    def __init__(self, *args, background_dir, background_probability=0.85, **kwargs):
        self.background_dir = str(Path(background_dir).resolve())
        self.background_probability = background_probability
        super().__init__(*args, **kwargs)
        (self.save_dir / "background_augmentation.json").write_text(json.dumps({
            "background_dir": self.background_dir, "probability": self.background_probability,
            "applies_to": "training only; validation uses original photos",
            "implementation": str(Path(__file__).resolve()),
        }, indent=2) + "\n")

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return make_training_dataset(img_path, self.data, self.args, batch,
                                     self.background_dir, self.background_probability, stride)
