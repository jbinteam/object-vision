#!/usr/bin/env python3
"""Compare CPU and CUDA YOLO on the same recorded RGB image, including transfers."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from object_pose_tracking.segmentation import YoloSegmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb', type=Path, required=True, help='H x W x 3 uint8 RGB .npy file')
    parser.add_argument('--weights', type=Path, default=Path('weights/best.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--half', action='store_true', help='Use FP16 for CUDA only')
    args = parser.parse_args()
    if not 1 <= args.iterations <= 200:
        parser.error('iterations must be in [1,200]')
    rgb = np.load(args.rgb, allow_pickle=False)
    models = {name: YoloSegmenter(str(args.weights), device=name, half=args.half and name != 'cpu')
              for name in ('cpu', 'cuda:0')}
    # The exact image dimensions warm both the model and postprocessing kernels.
    for model in models.values():
        for _ in range(3):
            model.predict(rgb)
    samples = {name: [] for name in models}
    detections = {}
    for index in range(args.iterations):
        names = list(models) if index % 2 == 0 else list(models)[::-1]
        for name in names:
            start = time.perf_counter()
            detections[name] = models[name].predict(rgb)
            samples[name].append((time.perf_counter() - start) * 1000)
    ious = []
    for cpu in detections['cpu']:
        candidates = []
        for gpu in detections['cuda:0']:
            union = np.count_nonzero(cpu.mask | gpu.mask)
            candidates.append(float(np.count_nonzero(cpu.mask & gpu.mask) / union) if union else 1.)
        ious.append(max(candidates, default=0.))
    report = dict(input=str(args.rgb), shape=list(rgb.shape), iterations=args.iterations,
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  cuda_half=args.half, detections={name: len(v) for name, v in detections.items()},
                  cpu_mask_best_cuda_ious=ious,
                  note='One input image measures timing and mask agreement, not dataset accuracy.',
                  inference_ms={name: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)))
                                for name, v in samples.items()})
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
