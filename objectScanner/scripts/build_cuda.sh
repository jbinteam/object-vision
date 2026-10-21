#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
if [[ ! -x "$project_root/.venv-ros-cuda/bin/python" ]]; then
    echo 'Create .venv-ros-cuda as documented in docs/CUDA_TRACKING.md first.' >&2
    exit 1
fi
cd "$project_root/ros2_ws"
"$project_root/.venv-ros-cuda/bin/python" -m colcon --log-base log-cuda build \
    --build-base build-cuda --install-base install-cuda --symlink-install \
    --packages-select object_pose_tracking
