"""Training-only background replacement shared by YOLO segmentation and pose.

The foreground image geometry is never changed here.  Consequently the same
segmentation polygon and ordered 2D keypoints remain valid for every sampled
background.  Validation always reads the original photographs unchanged.
"""

from functools import lru_cache
import json
from pathlib import Path
import random

import cv2
import numpy as np

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.torch_utils import unwrap_model


class BackgroundReplacer:
    def __init__(self, directory):
        directory = Path(directory).resolve()
        manifest = directory / "sources.json"
        if manifest.is_file():
            self.paths = [directory / row["file"]
                          for row in json.loads(manifest.read_text())["images"]]
        else:
            self.paths = sorted(
                path for path in directory.rglob("*")
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.paths or not all(path.is_file() for path in self.paths):
            raise ValueError(f"Backgrounds missing in {directory}")

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
        scale = max(width / background.shape[1], height / background.shape[0])
        scale *= rng.uniform(1.0, 1.3)
        resized = cv2.resize(
            background,
            (max(width, int(np.ceil(background.shape[1] * scale))),
             max(height, int(np.ceil(background.shape[0] * scale)))),
        )
        x = rng.randint(0, resized.shape[1] - width)
        y = rng.randint(0, resized.shape[0] - height)
        canvas = resized[y:y + height, x:x + width].copy()
        if rng.random() < 0.5:
            canvas = canvas[:, ::-1].copy()
        canvas[mask > 0] = image[mask > 0]
        return canvas


class BackgroundYOLODataset(YOLODataset):
    def __init__(self, *args, dataset_root, background_dir,
                 background_probability=0.85, epoch_repeats=1, **kwargs):
        self.dataset_root = Path(dataset_root).resolve()
        self.background_probability = float(background_probability)
        self.epoch_repeats = int(epoch_repeats)
        if self.epoch_repeats < 1:
            raise ValueError("epoch_repeats must be at least 1")
        self.background_replacer = BackgroundReplacer(background_dir)
        super().__init__(*args, **kwargs)
        self.mask_paths = {}
        for image in self.im_files:
            # Preserve the dataset-side path.  ``resolve()`` would follow the
            # exporter's space-saving image symlink back into the capture tree.
            relative = Path(image).absolute().relative_to(self.dataset_root / "images")
            mask = self.dataset_root / "masks" / relative.with_suffix(".png")
            if not mask.is_file():
                raise ValueError(f"Foreground mask missing: {mask}")
            self.mask_paths[image] = mask

    def __len__(self):
        """Draw each source repeatedly without materializing augmented images.

        A small capture set otherwise produces only a few optimizer steps per
        epoch. Repeating indices here retains the old materialized-dataset
        training volume while every read can receive a newly sampled background.
        """
        return len(self.im_files) * self.epoch_repeats

    def get_image_and_label(self, index):
        index %= len(self.im_files)
        label = super().get_image_and_label(index)
        if self.augment and random.random() < self.background_probability:
            mask = cv2.imread(str(self.mask_paths[label["im_file"]]), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != label["ori_shape"]:
                raise ValueError(f"Invalid foreground mask for {label['im_file']}")
            height, width = label["img"].shape[:2]
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            label["img"] = self.background_replacer.composite(label["img"], mask)
        return label


def make_training_dataset(img_path, data, hyp, batch, background_dir,
                          probability, epoch_repeats, task, stride=32):
    return BackgroundYOLODataset(
        img_path=img_path, imgsz=hyp.imgsz, batch_size=batch, augment=True,
        hyp=hyp, rect=hyp.rect, cache=None, single_cls=hyp.single_cls,
        stride=stride, pad=0.0, prefix="train: ", task=task, classes=hyp.classes,
        data=data, fraction=hyp.fraction, dataset_root=data["path"],
        background_dir=background_dir, background_probability=probability,
        epoch_repeats=epoch_repeats,
    )


class _BackgroundTrainerMixin:
    augmentation_task = None

    def __init__(self, *args, background_dir, background_probability=0.85,
                 epoch_repeats=1, **kwargs):
        self.background_dir = str(Path(background_dir).resolve())
        self.background_probability = float(background_probability)
        self.epoch_repeats = int(epoch_repeats)
        super().__init__(*args, **kwargs)
        (self.save_dir / "background_augmentation.json").write_text(json.dumps({
            "background_dir": self.background_dir,
            "probability": self.background_probability,
            "epoch_repeats": self.epoch_repeats,
            "applies_to": "training only; validation uses original photographs",
            "labels": "foreground geometry unchanged; segmentation and pose labels remain exact",
            "implementation": str(Path(__file__).resolve()),
        }, indent=2) + "\n")

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        stride = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return make_training_dataset(
            img_path, self.data, self.args, batch, self.background_dir,
            self.background_probability, self.epoch_repeats,
            self.augmentation_task, stride)


class BackgroundSegmentationTrainer(_BackgroundTrainerMixin, SegmentationTrainer):
    augmentation_task = "segment"


class BackgroundPoseTrainer(_BackgroundTrainerMixin, PoseTrainer):
    augmentation_task = "pose"
