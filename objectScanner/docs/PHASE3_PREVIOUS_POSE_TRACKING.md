# Phase 3: previous-pose fixed-reference tracking

Performance update: the default ICP backend now reuses a native CPU search index
within each fit. See [profiling, paired benchmarks, and configuration](TRACKING_PERFORMANCE.md).
An opt-in [CUDA version and hybrid preset](CUDA_TRACKING.md) are also available.
The CUDA environment passes 92 tests; the original Phase 3 measurements below
predate that optimization.

Phase 3 uses the last accepted **raw, absolute** `T_camera_object` as the initial
guess for the next ICP registration. Every registration still aligns the same
fixed object reference against the current observed cloud. It never aligns two
successive observations, accumulates corrections onto a previous pose, or changes
the reference's origin/axes.

## Run

```bash
./scripts/build_phase1.sh
./scripts/run_phase3.sh
```

The shared build helper builds all implemented phases. The Phase 3 launch starts
RGB-D capture, segmentation, colored cloud extraction, tracking, and RViz. Stop a
previous Phase 1/2 camera pipeline before starting this launch. Ctrl+C stops the
launch group. `rviz:=false` enables headless operation. No additional dependencies
were needed beyond Phase 2.

The RGB cloud, yellow transformed reference, and object-frame XYZ arrows use the
same [Phase 2 RViz configuration and transform conventions](PHASE2_FIXED_REFERENCE_ICP.md).
TF remains `camera_color_optical_frame → object`; the frame is the one defined by
the exported reference. Phase 2's launcher remains available for independently
initialized ICP comparisons.

All Phase 2 launch arguments remain available. For example:

```bash
./scripts/run_phase3.sh \
  reference:=/absolute/path/reference.ply \
  icp_params_file:=/absolute/path/tracking.yaml
```

The complete Phase 2 parameter file loads first, followed by the selected override.
The default override is
[`phase3.yaml`](../ros2_ws/src/object_pose_tracking/config/phase3.yaml):

```yaml
icp_pose:
  ros__parameters:
    tracking_enabled: true
    tracking_seed_timeout: 0.75
```

If supplying a custom override, keep `tracking_enabled: true` to enable Phase 3.
The initial orientation modes, quality checks, timing limits, and reference remain
those in Phase 2 unless explicitly overridden. Parameters are read-only at runtime.

## Seed lifecycle

1. Without a usable stored pose, initialize from the configured rotation hint and
   reference/observed centroids, as in Phase 2.
2. Run ICP from that seed, against the unchanged fixed reference.
3. Commit a **copy** of the accepted raw result only after validation, data-age
   checks, and output publication. Published results still use the measurement stamp.
4. The next observation receives a separate copy of that pose as its initial guess.
5. A rejected, invalid, or expired result cannot replace the stored pose. A brief
   missed/empty observation clears visual output but preserves the seed until it
   expires. No rejected or stale pose/TF is published.
6. The seed expires when its accepted measurement reaches `tracking_seed_timeout`.
   Rejections do not refresh that timestamp. Expiry is checked by a timer and again
   before selecting a seed, so continuously arriving failures cannot preserve it
   indefinitely. The next usable observation then initializes from centroids again.

The timeout starts at **measurement time**, including earlier camera/inference/ICP
latency, not at the end of processing. Backward clock movement invalidates the
stored seed. Delayed or duplicate observations cannot overwrite a newer pose.
The existing one-message pending observation queue and data-age limits remain.

This is seed memory, not the full Phase 4 state machine. Motion-jump rejection,
configurable consecutive-failure transitions, explicit LOST state, and stronger
recovery remain Phase 4. Pose filtering is Phase 5. A bad local minimum that passes
the current checks can still become the tracking seed; physical alignment must
be inspected. Warm starts do not solve the first-pose/global-orientation problem.

## Diagnostics and measurement

`/diagnostics` now includes:

- `initial_guess`: `centroid` or `previous_pose` for each registration attempt.
- `tracking_seed_age_ms`: age of the stored accepted measurement when selected.
- `translation_delta_m` and `rotation_delta_rad`: change from the seed for accepted
  previous-pose fits. These are measurements, not Phase 4 jump gates yet.
- `tracking_seed_expired`, plus the existing acceptance/rejection and timing fields.

To measure the live pipeline:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
.venv-ros/bin/python scripts/measure_phase2.py --seconds 20 \
  --output /tmp/phase3_live.json
```

The shared observer reports accepted/unaccepted seed modes separately in
`seed_modes`. It measures the current pipeline; its filename does not restrict it
to Phase 2. An actual Phase 3 live acquisition should show a centroid-initialized
acceptance followed by accepted `previous_pose` attempts.

## Verification

The ROS package build and all **62 tests** pass, including ten additions for Phase 3:
seed copy isolation, expiration, stale/invalid/duplicate updates, preservation after
rejection, absolute-pose replacement, point-to-point and point-to-plane motion
sequences, ROS handling of empty input/expired results, and Phase 2 compatibility.

```bash
./scripts/test_phase1.sh -q
```

A deterministic comparison used the same 30-frame full observation sequence of an
asymmetric synthetic surface for both modes. Each frame moves by 0.7 mm in X and
small roll/pitch/yaw increments; both use the same first centroid initialization.
The following medians exclude the first acquisition:

| ICP method | Independent median iterations / solver time | Previous-pose median iterations / solver time | Accepted frames (independent / previous-pose) |
| --- | --- | --- | --- |
| Point-to-point | 22 / 29.9 ms | 3 / 3.7 ms | 30/30 / 30/30 |
| Point-to-plane | 5 / 10.0 ms | 3 / 4.8 ms | 25/30 / 30/30 |

Phase 3 recovered every ground-truth transform in this exact, noiseless sequence
within 1e-5 m and 1e-3 rad, and tests verify the reference remains unchanged.
These synthetic results do not establish physical pose accuracy, occlusion
robustness, or live 10 Hz operation. Some independently initialized point-to-plane
candidates diverged and were rejected as the object rotated away from the seed.
The raw report includes rejected candidate errors, which are not valid output poses.

Raw results: [phase3_synthetic_20260923.json](validation/phase3_synthetic_20260923.json).
Reproduce the comparison without opening the camera or publishing ROS messages:

```bash
source ros2_ws/install/setup.bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-ros/bin/python \
  scripts/benchmark_tracking.py --frames 30 --output /tmp/phase3_synthetic.json
```

A later live 20.01-second check acquired the target and recorded:

| Measurement | Result |
| --- | --- |
| Accepted poses / axis updates | 70 / 70 |
| Accepted seeds from previous pose / centroid | 62 / 8 |
| Accepted output rate over this window | 3.50 Hz |
| Accepted ICP median / p95 | 52.8 / 130.6 ms |
| Accepted receipt-to-pose median / p95 | 244.2 / 352.2 ms |
| Rejected attempts | 42 iteration limits, 14 time limits |
| Seed expirations | 8 |

Raw results: [phase3_live_20260923.json](validation/phase3_live_20260923.json).
This verifies live use of previous accepted poses and automatic return to centroid
seeding after expiry. It does not establish physical pose accuracy or stable 10 Hz
tracking. The current live surface still causes convergence failures, and the
elapsed latency includes segmentation and other upstream work. Phase 2 and Phase 3
live windows used different object motions/views, so they are not a controlled
speed comparison.

Live first-pose acquisition and subsequent tracking must be assessed separately
from this synthetic comparison. If the current view cannot produce a valid initial
fit, restore a view close to the reference scan or configure a suitable initial
orientation; do not lower quality gates solely to make a frame appear.

## Changes

- `tracking.py`: small independent accepted-pose seed store.
- `pose_node.py`: optional previous-pose seeding and diagnostics; existing reference,
  ICP backend, pose, TF, and visual output contracts remain unchanged.
- `phase3.yaml`, `phase3.launch.py`, `scripts/run_phase3.sh`: tracking launch.
- `phase2.launch.py`: baseline parameters followed by optional overrides.
- `scripts/benchmark_tracking.py`: controlled comparison with ground truth.
- `scripts/measure_phase2.py`: seed-mode counts for live verification.
- Algorithm/ROS tests and documentation updated.
