#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_dir "$DATASET/reference_scans"
if ! compgen -G "$DATASET/reference_scans/*.ply" >/dev/null; then
    echo "No scan PLY found in $DATASET/reference_scans" >&2
    exit 1
fi

port="${REFERENCE_EDITOR_PORT:-8765}"
echo "Open http://127.0.0.1:$port/?reference_name=$CATEGORY&frame_id=box_center&lock_frame=1"
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/webui.py" \
    --port "$port" \
    --scan-dir "$DATASET/reference_scans" \
    --reference-dir "$DATASET/references" \
    --object-frame box_center --fixed-frame-from-scan
