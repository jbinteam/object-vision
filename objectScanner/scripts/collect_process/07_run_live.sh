#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_file "$SEG_WEIGHTS" "$POSE_WEIGHTS" "$ALIGNED_REFERENCE" \
    "$PNP_SNAPSHOT/calibration.json" "$PNP_SNAPSHOT/object_keypoints.yaml" \
    "$PNP_SNAPSHOT/manifest.json"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/pnp_snapshot.py" verify "$DATASET"
"$PROJECT_ROOT/.venv-ros-cuda/bin/python" \
    "$PROJECT_ROOT/scripts/verify_phase4_frame_contract.py" "$DATASET" "$CATEGORY" \
    --calibration "$PNP_SNAPSHOT/calibration.json" \
    --keypoints "$PNP_SNAPSHOT/object_keypoints.yaml"

exec "$PROJECT_ROOT/scripts/run_phase4_cuda.sh" \
    "camera_backend:=$CAMERA_BACKEND" \
    "segmentation_weights:=$SEG_WEIGHTS" \
    "pose_weights:=$POSE_WEIGHTS" \
    "reference:=$ALIGNED_REFERENCE" \
    "keypoints:=$PNP_SNAPSHOT/object_keypoints.yaml" \
    "${@:3}"
