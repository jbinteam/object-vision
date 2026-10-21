#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_dir "$DATASET" "$PROJECT_ROOT/../objectSegmentation/.venv"

cd "$PROJECT_ROOT"
exec ../objectSegmentation/.venv/bin/python -E scripts/augment_centerpose_backgrounds.py \
    --dataset "$DATASET" \
    --output "$DATASET/centerpose_masks" \
    "${@:3}"
