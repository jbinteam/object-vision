#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$project_root/ros2_ws/install-cuda/setup.bash"
export ROS_DOMAIN_ID=84
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
# Bound the complete test run, including native GPU calls.
exec timeout -k 5s 180s "$project_root/.venv-ros-cuda/bin/python" -m pytest \
    "$project_root/ros2_ws/src/object_pose_tracking/test" "$@"
