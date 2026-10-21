# Phase 2: fixed-reference ICP and the object frame

Phase 3 is now available separately: [previous-pose tracking](PHASE3_PREVIOUS_POSE_TRACKING.md).
This page describes the independent initialization retained by the Phase 2 launch.

Phase 2 implements independent pose estimation for each usable segmented cloud.
It loads the fixed reference once, initializes rotation/centroid translation, runs
CPU ICP, and publishes only results that pass the Phase 2 checks. RViz shows the
estimated object frame directly on the object and the aligned reference cloud.
Previous-pose seeding, the full tracking state machine, filtering, and robot
integration remain later phases.

## Run

From `/home/jb/codex/projects/objectScanner`:

```bash
.venv-ros/bin/python -m pip install -r requirements-ros.txt
./scripts/build_phase1.sh
./scripts/run_phase2.sh
```

The build helper builds the whole package, including Phase 2. RViz starts by
default. Use `rviz:=false` for headless operation. Stop the launch with Ctrl+C.
The launch starts camera, segmentation, cloud extraction, ICP, and optional RViz;
do not run Phase 1 concurrently against the same camera/topics.

The added dependency is `open3d-cpu==0.19.0` in the ROS Python 3.12 environment.
The existing Python 3.10 scanner/camera environment is unchanged. OpenMP and
OpenBLAS are limited to two threads for the ICP process. GPU compute is not used.

Equivalent ROS commands:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/jb/codex/projects/objectScanner/ros2_ws
../.venv-ros/bin/python -m colcon build --symlink-install \
  --packages-select object_pose_tracking
source install/setup.bash
../.venv-ros/bin/python /opt/ros/jazzy/bin/ros2 launch \
  object_pose_tracking phase2.launch.py \
  project_root:=/home/jb/codex/projects/objectScanner
```

Example using other assets/configuration:

```bash
./scripts/run_phase2.sh \
  reference:=/absolute/path/reference.ply \
  weights:=/absolute/path/segmentation.pt \
  icp_params_file:=/absolute/path/phase2.yaml
```

`params_file` configures the Phase 1 nodes; `icp_params_file` configures ICP.
`camera:=false` accepts an external aligned RGB-D source with the Phase 1 topic
and frame contracts. Parameters are read-only at runtime; restart to apply edits.

## What RViz shows

- **Segmented object:** the current observed cloud with RGB colors.
- **Aligned reference (ICP):** yellow points from the fixed reference transformed
  by the accepted pose. This should overlap the observed surface; hidden reference
  surfaces may have no counterpart in the visible observation.
- **ICP object frame (XYZ):** arrows at the estimated reference-frame origin:
  red X, green Y, blue Z, with an `object` label. Default arrow length is 0.06 m.
- **Move Camera / Orbit:** left-drag rotates, middle-drag pans, scroll/right-drag
  zooms. Focus Camera centers the view on a clicked point.

The arrows represent the frame chosen when exporting the reference, not a newly
computed centroid/PCA frame. If the chosen origin lies inside the object, the
arrows start inside it. A rejected or expired estimate clears the arrows and
overlay. Marker lifetimes also expire if the node stops publishing.

## Topics and frame convention

| Interface | Type / meaning |
| --- | --- |
| `/object/cloud` | Input `PointCloud2`, XYZ in metres, optional RGB |
| `/object/pose` | Accepted `geometry_msgs/PoseStamped` |
| `/tf` | Accepted `camera_color_optical_frame → object` |
| `/object/reference_cloud_transformed` | Reference points transformed into the camera frame |
| `/object/frame_axes` | Expiring `visualization_msgs/MarkerArray` arrows and label |
| `/diagnostics` | Acceptance/rejection reason, fit metrics, preparation/ICP/total timing |

The current parent is the optical frame because back-projection produces optical
coordinates. To use another parent, first configure the cloud extractor to apply
its calibrated TF and use that same `camera_frame` for ICP. A cloud with another
frame ID is rejected; it is never merely relabeled.

`T_camera_object` means `p_camera = T_camera_object @ p_object` for homogeneous
column vectors. The reference remains permanently in the exported object frame.
The configured ROS child name `object` names that same frame (export metadata calls
it `obj_frame`); no basis or origin is changed. Pose, TF, axes, and each reference
overlay use the same accepted transform and the original cloud measurement stamp.
The underlying stamp is the host receipt time described in Phase 1.

A stale or rejected candidate never emits a new pose/TF, and old timestamps are
never refreshed. TF consumers can still retain previous transforms in their
caches: check age and diagnostics. This is a pose-estimation checkpoint, not a
robot-control validity contract or the full Phase 4 LOST state machine.

## Reference and initialization

Default reference: `references/emergency_stop_ref/reference.ply`, 3,194 original
points in metres. The file is loaded and voxel-downsampled once at startup.
The permanent cloud is never transformed in place; temporary copies alone are
used for the reference overlay. `.ply` and `.pcd` are accepted, assuming metres.

Available `initial_rotation_mode` values:

| Mode | Rotation seed |
| --- | --- |
| `identity` | Identity rotation; useful when object axes already approximately match camera axes |
| `manual` | `initial_rotation_deg: [roll, pitch, yaw]`, with Rz(yaw) Ry(pitch) Rx(roll) |
| `reference_metadata` | Rotation only from `T_scan_obj` in `frame.json` beside the reference |

The supplied Phase 2 YAML uses `reference_metadata`, which is an orientation hint
for a view similar to the original reference scan. It **does not** apply the export
transform to the reference a second time, and it does not reuse the old scan
translation. Each observation starts with:

```text
R0 = configured orientation hint
t0 = centroid(observed) - R0 @ centroid(fixed_reference)
```

For a different viewpoint, choose an appropriate manual seed, or identity when
appropriate. Another reference without `frame.json` needs identity/manual mode.
Missing/invalid required metadata fails at startup rather than silently changing
the initialization. The initialization function and registration backend are
separate so coarse global registration can be added later.

Each callback currently starts independently from this seed. The last pose is
stored only for output expiration; it is not reused as the next ICP seed until
Phase 3. Strong occlusion, wrong orientation, or symmetry can produce a local
minimum even with acceptable fit scores. Inspect the axes and overlay before
accepting the physical interpretation of the result.

## Example parameters and ICP checks

The complete example is
[`phase2.yaml`](../ros2_ws/src/object_pose_tracking/config/phase2.yaml):

```yaml
icp_pose:
  ros__parameters:
    camera_frame: camera_color_optical_frame
    object_frame: object
    icp_type: point_to_plane
    voxel_size: 0.003
    icp_max_iterations: 60
    icp_max_correspondence_distance: 0.025
    icp_transformation_epsilon: 0.00001
    icp_rotation_epsilon: 0.0001
    icp_time_budget: 0.25
    normal_radius: 0.015
    normal_max_neighbors: 30
    fitness_threshold: 0.35
    rmse_threshold: 0.012
    minimum_points: 30
    minimum_correspondences: 30
    initial_rotation_mode: reference_metadata
    initial_rotation_deg: [0.0, 0.0, 0.0]
    maximum_data_age: 0.75
    axis_length: 0.06
    visualization_lifetime: 0.5
    reference_publish_rate: 10.0
```

Distances are metres and angular convergence is radians; only manual seed angles
use degrees. `point_to_point` is also supported. Point-to-plane estimates normals
on the current **target** cloud, since that is what Open3D's method uses. Source
normal orientation from the scanner does not control these target normals.

The adapter runs bounded single ICP iterations and examines the actual change
between absolute source-to-target transforms. `converged` means both incremental
translation and rotation are below their thresholds. Hitting the iteration/time
limit is not called convergence. A synchronous native ICP iteration can exceed
the time budget; its result is rejected afterward, but the budget cannot interrupt
a native call or recover a hung driver/library.

Additional checks require finite rigid transforms, enough nondegenerate points,
sufficient correspondences, fitness, and RMSE. Point-to-plane also rejects a
rank-deficient six-dimensional correspondence system, such as a single featureless
plane whose tangential pose cannot be constrained. There is no silent fallback to
point-to-point; choose that method explicitly if appropriate.

Diagnostics report Open3D fitness (matched source fraction), Euclidean inlier RMSE,
correspondence count, iteration count, termination reason, and timing. The reported
correspondence ratio currently equals fitness; it is not an independent confidence
measure. Thresholds are initial configuration values, not calibrated object-pose
accuracy guarantees. Full jump rejection and repeated-failure state handling remain
Phase 4.

The observed-cloud subscription retains one pending message. Old measurements and
results that expire during ICP are rejected. The debug reference rate can be lowered
without changing pose publication; each overlay retains its own accepted pose stamp.

## Tests and measured status

```bash
./scripts/test_phase1.sh -q
source ros2_ws/install/setup.bash
cd ros2_ws
ROS_DOMAIN_ID=83 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  ../.venv-ros/bin/python -m colcon test --packages-select object_pose_tracking
../.venv-ros/bin/python -m colcon test-result --verbose
```

The shared test helper now runs all 52 package tests, including 22 added for Phase 2:

- Identity, translation, rotation, and combined transforms with both ICP methods.
- Noisy and partial observations with controlled initial guesses.
- Absolute transform direction, source immutability, metadata rotation seeding.
- Invalid clouds, no correspondences, bad initial transforms, iteration/time limits,
  high RMSE, and underconstrained planar geometry.
- ROS pose/TF/axis/overlay consistency, original timestamps, empty-input clearing,
  stale input, timeout clearing without refreshed pose/TF, and new observations.

On the deterministic asymmetric full-surface tests, both methods satisfy translation
error below 0.00001 m and rotation error below 0.001 rad. Noisy/partial tests satisfy
0.003 m and 0.06 rad with controlled seeds. These are synthetic test tolerances,
not measured accuracy for the physical emergency stop.

The first 15-second Phase 2 hardware check received 151 empty observed clouds and
published no pose, correctly withholding the frame when no object was detected.
Later in the same session, live detections produced accepted ICP poses interspersed
with rejected fits (correspondence, conditioning, RMSE, and convergence limits).
This confirms the live pose path executes, but stable accepted-pose rate, reference
overlap, and physical frame accuracy remain unverified. RViz loaded the observed
cloud, reference, and frame displays successfully. The initial no-target probe is
saved in [phase2_live_20260923.json](validation/phase2_live_20260923.json).

A subsequent 20.04-second window, including startup and intervals with no detection,
recorded **46 accepted poses and 46 axis-marker updates**. It also recorded 20 rejected
ICP attempts (3 insufficient correspondences, 13 iteration limits, 4 time limits).
Accepted results had median ICP execution 70.3 ms, median receipt-to-pose latency
257.9 ms (p95 370.6 ms), and median inlier RMSE 5.50 mm. The 2.30 accepted poses/s
across that mixed window is not a steady-state tracking-rate measurement. Registration
remains intermittent; fit RMSE is not physical pose accuracy. Results are saved in
[phase2_live_poses_20260923.json](validation/phase2_live_poses_20260923.json).


To measure with an object in view:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
.venv-ros/bin/python scripts/measure_phase2.py --seconds 15 \
  --output /tmp/phase2_metrics.json
```

The observer separates accepted and unaccepted timing samples. `icp_ms` measures
the iterative solver; `preparation_ms` includes target construction/downsampling,
normal estimation, and seed generation; `pose_total_ms` covers the callback;
`receipt_to_pose_ms` includes the earlier RGB-D and segmentation stages.
The existing Phase 1 latency already exceeded 100 ms in its initial test, so 10 Hz
pose throughput and sub-100-ms end-to-end latency are not established here.

Next steps: verify the physical frame and overlay in RViz, then Phase 3 can seed
from the last accepted pose. Profile target normal estimation, independent
initialization, and stage latency before tuning voxel size, correspondences,
inference resolution, or thread counts. Do not lower quality gates solely to
make a frame appear.

## Files

New algorithm/output modules: `registration.py`, `pose_node.py`, `pose_outputs.py`.
New launch/config/view: `phase2.launch.py`, `phase2.yaml`, `phase2.rviz`.
New helpers: `scripts/run_phase2.sh`, `scripts/measure_phase2.py`.
New tests: `test_registration.py`, `test_pose_node.py`.
Package entrypoints/dependencies, ROS requirements, and documentation are updated.
Phase 1 remains available with its original launch.

Backend references: [Open3D ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
and [CPU package installation](https://www.open3d.org/docs/0.17.0/getting_started.html).
