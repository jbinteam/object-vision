#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

cd "$PROJECT_ROOT"
exec .venv/bin/python scripts/capture_turntable_dataset.py \
    --name "$SESSION" \
    --camera-backend "$CAMERA_BACKEND" \
    --centerpose-category "$CATEGORY" \
    --centerpose-split train \
    --tag-family tagStandard41h12 \
    --setup-tag-size-mm 14 \
    --permanent-tag-size-mm 28 \
    --setup-ids 1,2,3,4 \
    --permanent-ids 5,6,7,8,9,10 \
    --scan-device cuda \
    --scan-downscale 1 \
    --scan-stall-timeout 60 \
    "${@:3}"
