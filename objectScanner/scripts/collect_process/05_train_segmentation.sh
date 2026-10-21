#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_file "$YOLO_DATASET/segmentation/data.yaml"
require_dir "$PROJECT_ROOT/../objectSegmentation/backgrounds/picsum"
"$PROJECT_ROOT/.venv-ros-cuda/bin/python" \
    "$PROJECT_ROOT/scripts/verify_phase4_frame_contract.py" "$DATASET" "$CATEGORY"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/pnp_snapshot.py" save "$DATASET"

cd "$PROJECT_ROOT"
exec ../objectSegmentation/.venv/bin/python -E scripts/train_yolo_seg_pose.py \
    --dataset "$YOLO_DATASET" \
    --project "$PROJECT_ROOT/training/yolo" \
    --name "${SESSION}_onthefly" \
    --task segmentation \
    --epoch-repeats 12 \
    --backgrounds "$PROJECT_ROOT/../objectSegmentation/backgrounds/picsum" \
    "${@:3}"
