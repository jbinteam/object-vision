#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
if [[ ! -x "$project_root/.venv-ros/bin/python" ]]; then
    echo 'Create .venv-ros as documented in docs/PHASE1_RGBD_SEGMENTATION.md first.' >&2
    exit 1
fi
cd "$project_root/ros2_ws"
"$project_root/.venv-ros/bin/python" -m colcon build --symlink-install --packages-select object_pose_tracking
