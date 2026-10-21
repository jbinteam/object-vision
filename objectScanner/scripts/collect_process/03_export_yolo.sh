#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_dir "$DATASET/centerpose_masks/masks"

cd "$PROJECT_ROOT"
exec .venv/bin/python scripts/generate_yolo_seg_pose_dataset.py \
    --dataset "$DATASET" \
    --augmentation-root "$DATASET/centerpose_masks" \
    --output "$YOLO_DATASET" \
    "${@:3}"
