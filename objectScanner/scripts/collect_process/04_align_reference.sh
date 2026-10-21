#!/usr/bin/env bash
set -euo pipefail
echo "04_align_reference.sh now builds a calibrated RGB-D reference." >&2
exec "$(dirname -- "${BASH_SOURCE[0]}")/04_build_reference.sh" "$@"
