# Pinned upstream dependencies

Submodules pin source identities; initialize them with
`GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive` from the repository root.
This skips optional large upstream LFS payloads. Fetch them explicitly inside the
relevant submodule if an Isaac experiment requires them.

| Location | Upstream | Commit | Use |
| --- | --- | --- | --- |
| `objectSegmentation/BiRefNet` | https://github.com/ZhengPeng7/BiRefNet.git | `ebcc0bc8ec7fe919cec829f2dea656b3078acddc` | Foreground extraction; required for mask generation |
| `objectScanner/isaac_ros_ws/src/isaac-ros-cli` | https://github.com/NVIDIA-ISAAC-ROS/isaac-ros-cli.git | `680c8c5f85854bee8b0074a00a17c2ba2aeab906` | Historical optional Isaac ROS experiments |
| `objectScanner/isaac_ros_ws/src/isaac_ros_pose_estimation` | https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation.git | `0df2f66639e6ed65dafbc06264860b0cc29a5829` | Historical optional Isaac ROS experiments |

Current Phase 4 uses the repository's `object_pose_tracking` ROS package; Isaac ROS
is not required for that path. Custom Isaac Docker/tool files and workspace config
are retained; downloaded Isaac assets, engines and ROS bags are excluded. The two
workspace config files in `objectScanner/isaac_ros_ws/src/` can also be copied into
the `isaac-ros-cli/` directory if reproducing the original CLI experiment setup.

## Optional historical CenterPose workflow

The CenterPose launcher expects a sibling of `objectScanner/`, or an explicit
`--centerpose-root`. If resuming that workflow, from this repository root:

```bash
git clone https://github.com/NVlabs/CenterPose.git CenterPose
git -C CenterPose checkout 6c89d420b33bd01c14c13f509af08bfe3d8b2fe7
git -C CenterPose apply ../objectScanner/patches/centerpose-modern-pytorch.patch
```

That patch preserves the original local compatibility changes to DCNv2 and the
trackers. Follow the scanner's CenterPose environment instructions separately.
The upstream checkout is excluded from this repository; retain and review its
own license when using it. It is not needed for current YOLO/PnP/GICP tracking.

Python dependencies remain in each project's requirement files, with separate
camera/ROS/CPU/CUDA environments. The original workstation's permissive Codex
execution settings were omitted; named scanner agent configurations remain as
optional development tooling, not runtime requirements.
