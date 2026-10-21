#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_file "$REFERENCE" "$DATASET/references/$CATEGORY/frame.json"

port="${REFERENCE_EDITOR_PORT:-8765}"
echo "Open http://127.0.0.1:$port/?frame_id=box_center&lock_frame=1&crop_only=1&reference_name=$CATEGORY"
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/webui.py" \
    --port "$port" \
    --production-reference-dir "$DATASET/references/$CATEGORY"
