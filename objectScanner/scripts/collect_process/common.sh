#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${1:-${COLLECT_SESSION:-my_object_gemini305}}"
if [[ -z "$SESSION" || "$SESSION" == */* || "$SESSION" == .* ]]; then
    echo "Usage: $0 [SESSION] [CATEGORY] [extra arguments...]" >&2
    echo "SESSION must be a simple name under datasets/." >&2
    exit 2
fi
DEFAULT_CATEGORY="$SESSION"
if [[ "$SESSION" == my_object_gemini305 || "$SESSION" == my_object_fresh || "$SESSION" == my_object_20260925 || "$SESSION" == my_object_20260925_r2 ]]; then
    DEFAULT_CATEGORY=my_object
fi
CATEGORY="${2:-${COLLECT_CATEGORY:-$DEFAULT_CATEGORY}}"
if [[ -z "$CATEGORY" || "$CATEGORY" == */* || "$CATEGORY" == .* ]]; then
    echo "CATEGORY must be a simple name." >&2
    exit 2
fi
CAMERA_BACKEND="${COLLECT_CAMERA_BACKEND:-orbbec}"
DATASET="$PROJECT_ROOT/datasets/$SESSION"
YOLO_DATASET="$DATASET/yolo_seg_pose_onthefly"
SEG_WEIGHTS="$PROJECT_ROOT/training/yolo/${SESSION}_onthefly_seg/weights/best.pt"
POSE_WEIGHTS="$PROJECT_ROOT/training/yolo/${SESSION}_pose_refined_pose/weights/best.pt"
REFERENCE="$DATASET/references/$CATEGORY/reference.ply"
ALIGNED_REFERENCE="$DATASET/references/$CATEGORY/reference_aligned.ply"
if [[ ! -f "$ALIGNED_REFERENCE" ]]; then
    ALIGNED_REFERENCE="$REFERENCE"
fi
KEYPOINTS="$YOLO_DATASET/object_keypoints.yaml"
PNP_SNAPSHOT="$DATASET/training_pnp"

require_file() {
    local path
    for path in "$@"; do
        if [[ ! -f "$path" ]]; then
            echo "Missing file: $path" >&2
            exit 1
        fi
    done
}

require_dir() {
    local path
    for path in "$@"; do
        if [[ ! -d "$path" ]]; then
            echo "Missing directory: $path" >&2
            exit 1
        fi
    done
}
