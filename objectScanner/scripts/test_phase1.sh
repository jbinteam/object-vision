#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$project_root/ros2_ws/install/setup.bash"
# Isolate synthetic topic publishers from the user's running camera pipeline.
export ROS_DOMAIN_ID=83
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec "$project_root/.venv-ros/bin/python" -m pytest "$project_root/ros2_ws/src/object_pose_tracking/test" "$@"
