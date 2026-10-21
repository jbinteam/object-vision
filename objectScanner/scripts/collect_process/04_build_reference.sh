#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_dir "$DATASET/centerpose_masks/masks"

cd "$PROJECT_ROOT"
exec .venv-ros-cuda/bin/python scripts/build_calibrated_reference.py \
    --dataset "$DATASET" \
    --masks "$DATASET/centerpose_masks" \
    --output "$DATASET/references/$CATEGORY" \
    "${@:3}"
