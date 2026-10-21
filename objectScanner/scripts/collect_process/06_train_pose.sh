#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_file "$YOLO_DATASET/pose/data.yaml"
POSE_START_WEIGHTS="${POSE_START_WEIGHTS:-$PROJECT_ROOT/training/yolo/my_object_rgbd_unfrozen_pose/weights/best.pt}"
require_file "$POSE_START_WEIGHTS"
"$PROJECT_ROOT/.venv-ros-cuda/bin/python" \
    "$PROJECT_ROOT/scripts/verify_phase4_frame_contract.py" "$DATASET" "$CATEGORY"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/pnp_snapshot.py" save "$DATASET"

cd "$PROJECT_ROOT"
exec ../objectSegmentation/.venv/bin/python -E scripts/train_yolo_seg_pose.py \
    --dataset "$YOLO_DATASET" \
    --project "$PROJECT_ROOT/training/yolo" \
    --name "${SESSION}_pose_refined" \
    --task pose \
    --pose-model "$POSE_START_WEIGHTS" \
    --epochs 100 --patience 25 --freeze 0 --lr0 0.00005 --epoch-repeats 12 \
    "${@:3}"
