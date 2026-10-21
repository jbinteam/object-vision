#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$project_root/ros2_ws/install/setup.bash"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec "$project_root/.venv-ros/bin/python" /opt/ros/jazzy/bin/ros2 launch \
    object_pose_tracking phase4.launch.py "project_root:=$project_root" "$@"
