"""Replaceable segmentation backend and conservative single-instance selection."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Detection:
    mask: np.ndarray
    box: np.ndarray
    confidence: float


@dataclass
class PoseDetection:
    box: np.ndarray
    confidence: float
    # N x 4: u, v, confidence, valid. Ordering is the YOLO data.yaml ordering.
    keypoints: np.ndarray


def box_iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    intersection = np.maximum(hi - lo, 0).prod()
    union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def associate_pose(segmentation, poses, minimum_iou=0.1):
    """Associate independently inferred segmentation and pose instances."""
    if segmentation is None:
        return None
    scored = [(box_iou(segmentation.box, pose.box), pose) for pose in poses]
    score, pose = max(scored, key=lambda item: item[0], default=(0., None))
    return pose if score >= minimum_iou else None


class InstanceSelector:
    """Acquire highest confidence; follow overlapping box; release after misses.

    This is spatial association, not persistent identity. Reacquisition can select
    another same-class object and is explicitly reported by the node.
    """
    def __init__(self, minimum_iou=0.1, reset_after_misses=5):
        if not 0 < minimum_iou <= 1 or reset_after_misses < 1:
            raise ValueError('Invalid association parameters')
        self.minimum_iou, self.reset_after_misses = minimum_iou, reset_after_misses
        self.reset()

    def reset(self):
        self.box, self.misses = None, 0

    def select(self, detections):
        if self.box is None:
            candidate = max(detections, key=lambda d: d.confidence, default=None)
            reason = 'acquired'
        else:
            candidates = [(box_iou(self.box, d.box), d) for d in detections]
            score, candidate = max(candidates, key=lambda item: item[0], default=(0, None))
            if score < self.minimum_iou:
                candidate = None
            reason = 'associated'
        if candidate is None:
            if self.box is None:
                return None, 'no_matching_instance'
            self.misses += 1
            if self.misses >= self.reset_after_misses:
                self.reset()
                return None, 'association_lost'
            return None, 'no_matching_instance'
        self.box, self.misses = candidate.box.copy(), 0
        return candidate, reason


class YoloSegmenter:
    def __init__(self, model_path, target_class='emergency_stop', confidence=0.5,
                 device='cpu', image_size=640, cpu_threads=2, half=False):
        if not Path(model_path).is_file():
            raise ValueError(f'Model file does not exist: {model_path}')
        if not 0 < confidence <= 1 or image_size < 32 or cpu_threads < 1:
            raise ValueError('Invalid inference parameters')
        import torch
        device = str(device).lower()
        if device != 'cpu':
            index = device.removeprefix('cuda:')
            if not index.isdecimal():
                raise ValueError('yolo_device must be cpu, cuda:N, or a GPU index')
            if not torch.cuda.is_available() or int(index) >= torch.cuda.device_count():
                raise RuntimeError(f'PyTorch CUDA:{index} is unavailable; use the CUDA environment '
                                   'or explicitly select yolo_device=cpu')
            device = f'cuda:{int(index)}'
        if half and device == 'cpu':
            raise ValueError('yolo_half requires a CUDA device')
        self.half, self._torch = bool(half), torch
        torch.set_num_threads(cpu_threads)
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='segment')
        if self.model.task != 'segment':
            raise ValueError('A segmentation checkpoint is required')
        names = self.model.names
        matches = [int(i) for i, name in names.items() if name == str(target_class) or str(i) == str(target_class)]
        if len(matches) != 1:
            raise ValueError(f'Target class {target_class!r} not unique in {names}')
        self.class_id = matches[0]
        self.confidence, self.device, self.image_size = confidence, device, image_size

    def predict(self, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected H x W x 3 uint8 RGB image')
        # Ultralytics numpy inputs are BGR. Retina masks have original H x W.
        result = self.model.predict(np.ascontiguousarray(rgb[..., ::-1]),
                                    classes=[self.class_id], conf=self.confidence,
                                    device=self.device, imgsz=self.image_size,
                                    half=self.half, retina_masks=True, verbose=False)[0]
        if self.device != 'cpu':
            self._torch.cuda.synchronize(self.device)
        if result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy() > 0.5
        if masks.shape[1:] != rgb.shape[:2]:
            raise ValueError('Backend returned masks that do not match the source image')
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [Detection(mask, box, float(score)) for mask, box, score in zip(masks, boxes, scores)]


class YoloPoseEstimator:
    """Ultralytics pose adapter that preserves deterministic keypoint order."""
    def __init__(self, model_path, target_class='my_object', confidence=0.25,
                 device='cpu', image_size=640, cpu_threads=2, half=False,
                 expected_keypoints=8):
        if not Path(model_path).is_file():
            raise ValueError(f'Pose model file does not exist: {model_path}')
        if not 0 < confidence <= 1 or expected_keypoints < 4:
            raise ValueError('Invalid pose inference parameters')
        import torch
        device = str(device).lower()
        if device != 'cpu':
            index = device.removeprefix('cuda:')
            if not index.isdecimal() or not torch.cuda.is_available() or int(index) >= torch.cuda.device_count():
                raise RuntimeError(f'PyTorch CUDA:{index} is unavailable')
            device = f'cuda:{int(index)}'
        if half and device == 'cpu':
            raise ValueError('yolo_half requires a CUDA device')
        torch.set_num_threads(cpu_threads)
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='pose')
        if self.model.task != 'pose':
            raise ValueError('A pose checkpoint is required')
        matches = [int(i) for i, name in self.model.names.items()
                   if name == str(target_class) or str(i) == str(target_class)]
        if len(matches) != 1:
            raise ValueError(f'Target pose class {target_class!r} not unique in {self.model.names}')
        self.class_id, self.confidence = matches[0], confidence
        self.device, self.image_size = device, image_size
        self.half, self.expected_keypoints, self._torch = bool(half), expected_keypoints, torch

    def predict(self, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected H x W x 3 uint8 RGB image')
        result = self.model.predict(np.ascontiguousarray(rgb[..., ::-1]),
                                    classes=[self.class_id], conf=self.confidence,
                                    device=self.device, imgsz=self.image_size,
                                    half=self.half, verbose=False)[0]
        if self.device != 'cpu':
            self._torch.cuda.synchronize(self.device)
        if result.keypoints is None or result.boxes is None:
            return []
        xy = result.keypoints.xy.cpu().numpy()
        confidence = result.keypoints.conf
        confidence = (np.ones(xy.shape[:2], dtype=np.float32) if confidence is None
                      else confidence.cpu().numpy())
        if xy.shape[1] != self.expected_keypoints:
            raise ValueError(f'Pose model returned {xy.shape[1]} keypoints; expected {self.expected_keypoints}')
        valid = np.isfinite(xy).all(axis=2) & np.isfinite(confidence) & np.any(xy != 0, axis=2)
        keypoints = np.concatenate((xy, confidence[..., None], valid[..., None]), axis=2)
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [PoseDetection(box, float(score), points.astype(np.float32))
                for box, score, points in zip(boxes, scores, keypoints)]
