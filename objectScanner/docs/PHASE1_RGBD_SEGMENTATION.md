# Phase 1: RGB-D segmentation and object cloud

Phase 2 is now available separately: [fixed-reference ICP and object-frame display](PHASE2_FIXED_REFERENCE_ICP.md).

Implemented on 2026-09-23. The ROS2 package builds, algorithm/ROS tests pass, and
live L515 capture with the supplied segmentation checkpoint works. Live target
cloud and RViz acceptance remain pending: the object was not detected during the
bounded hardware run. No pose estimation, reference registration, TF object pose,
or robot movement is implemented in this phase.

Gemini 305 is now the default acquisition backend. The earlier live results above
refer to L515; Gemini end-to-end tracking acceptance is a separate check. See
[Gemini 305 setup and calibration](GEMINI305_CAMERA.md).

## Run on this machine

From `/home/jb/codex/projects/objectScanner`:

```bash
./scripts/build_phase1.sh
./scripts/run_phase1.sh rviz:=true
```

Place the emergency stop in the camera view, approximately 0.3–1 m away. Close any
other application holding the camera. Stop with Ctrl+C; the launch group stops
all nodes and the camera worker. Omit `rviz:=true` for headless operation. RViz
uses software rendering to preserve the project's conservative GPU defaults.

The launch helper chooses these existing assets:

- Segmentation: `weights/best.pt`, Ultralytics 8.3.253, class `0: emergency_stop`.
- Camera interpreter: `.venv/bin/python` (Python 3.10 and the selected camera SDK).
- ROS/inference interpreter: `.venv-ros/bin/python` (system Python 3.12).
- The fixed reference `references/emergency_stop_ref/reference.ply` is reserved
  for Phase 2. Its 3,194 points are already in `obj_frame`, in metres. Do not
  apply `frame.json`'s scan-to-object transform to the reference again.

Custom launch inputs:

```bash
./scripts/run_phase1.sh weights:=/absolute/path/model.pt \
  params_file:=/absolute/path/phase1.yaml rviz:=true \
  camera_backend:=orbbec camera_serial:=YOUR_SERIAL
```

Parameters are loaded at startup and are read-only; restart after changing the
YAML. The checked-in example is
`ros2_ws/src/object_pose_tracking/config/phase1.yaml`.

Equivalent explicit ROS build/launch commands:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/jb/codex/projects/objectScanner/ros2_ws
../.venv-ros/bin/python -m colcon build --symlink-install \
  --packages-select object_pose_tracking
source install/setup.bash
../.venv-ros/bin/python /opt/ros/jazzy/bin/ros2 launch \
  object_pose_tracking phase1.launch.py \
  project_root:=/home/jb/codex/projects/objectScanner rviz:=true
```

## Environment and dependencies

Jazzy's installed Python extensions use Python 3.12. Camera acquisition remains
in `.venv` with Python 3.10 and `pyorbbecsdk2==2.1.2` (import `pyorbbecsdk`).
The optional L515 backend retains RealSense 2.54.2 in that environment. Camera
and ROS processes communicate through a length-bounded, non-pickle RGB-D pipe.

The installed ROS dependencies are declared in `package.xml`: rclpy,
message_filters, cv_bridge, sensor_msgs and sensor_msgs_py, diagnostic_msgs,
tf2_ros, launch_ros, ament_index_python, and RViz2. Build/test tools are colcon,
ament_python, setuptools, and pytest. NumPy and SciPy provide geometry and
statistical outlier removal. Phase 1 does not require Open3D in the ROS venv.

To recreate the additional environment after installing ROS2 Jazzy and those
ROS dependencies:

```bash
/usr/bin/python3 -m venv --system-site-packages .venv-ros
.venv-ros/bin/python -m pip install \
  'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
.venv-ros/bin/python -m pip install -r requirements-ros.txt
./scripts/build_phase1.sh
```

These are CPU-only PyTorch wheels. The original `.venv`, scanner dependencies,
and user-level PyTorch installation are unchanged. Keep `.venv/bin/python` for
scanner/reference-editor commands, and `.venv-ros/bin/python` for this ROS package.
Build with the explicit ROS venv interpreter so installed node entrypoints use it.

## Nodes, frames, and synchronization

| Node | Inputs | Outputs |
| --- | --- | --- |
| `rgbd_camera` | Isolated RGB-D worker | `/camera/color/image_raw`, `/camera/depth/image_raw`, `/camera/camera_info` |
| `yolo_segmentation` | Color image | `/object/mask` (`mono8`, 0 or 255) |
| `object_cloud` | Mask, RGB, aligned depth, CameraInfo | `/object/cloud` (`PointCloud2`, XYZ metres + packed RGB) |
| All nodes | Stage state | `/diagnostics` (`DiagnosticArray`) |

Despite the conventional `image_raw` topic names, the adapter publishes RGB and
aligned depth after the existing camera's rectification. CameraInfo uses the
corresponding adjusted intrinsics and zero distortion. RGB, depth, and calibration
share one header. The mask preserves that header through inference. Cloud extraction
synchronizes all four inputs, including the original RGB image. Each valid depth
point receives the color of the corresponding RGB pixel; voxel colors are averaged
and outlier removal preserves point/color pairing. The cloud contains `x`, `y`, `z`
float32 fields and a packed uint32 `rgb` field (`0x00RRGGBB`). Depth is
16UC1 millimetres; the cloud node also accepts 32FC1 metres from another source.

The timestamp is **host wall-clock frame receipt time before alignment and image
processing**, not sensor exposure time or inference-completion time. The worker
and ROS nodes run on the same machine. Live capture requires `use_sim_time=false`.
Timing diagnostics begin at that receipt timestamp, so true exposure-to-output
latency can be higher. Exact synchronization is the default; use a nonzero
`sync_slop` only for a source whose acquisition timestamps require it.

Back-projection produces optical coordinates: X right, Y down, Z forward.
Therefore the correct default output frame is `camera_color_optical_frame`.
This implements the plan's optical-frame clarification; it does not relabel these
points as body-frame `camera_link`. Set `camera_frame` to another frame only when
TF provides its calibrated transform from the input optical frame at the depth
measurement timestamp. Missing TF produces an empty cloud and an error diagnostic.
The adapter does not invent a body-frame extrinsic.

The reference's fixed-frame and `T_camera_object` conventions remain unchanged;
Phase 2 must use the chosen cloud parent frame consistently for pose and TF.

## Data selection and quality

- Initial selection is the highest-confidence detection of the configured class.
  Later selections require bounding-box overlap with the prior instance, with
  configurable minimum IoU. A higher-confidence distant instance cannot take over.
- Five unmatched observations release the association by default. Stalled input
  also releases it. Reacquisition can select another same-class object; this is
  spatial association, not a persistent object identity guarantee. Diagnostics
  distinguish acquisition, association, misses, and lost association.
- Empty masks are published when there is no detection. Empty clouds clear unusable
  observations instead of reusing an old cloud. Invalid depth, nonfinite values,
  out-of-range pixels, incompatible calibration, and undersized clouds are rejected.
- Optional voxel centroids and statistical outlier removal operate in metres.
  Set `voxel_size: 0.0` or `outlier_neighbors: 0` to disable either stage.
  Minimum/maximum depth, minimum points, voxel size, and outlier settings are in YAML.
- Camera transport has one replaceable latest-frame slot. Segmentation queues only
  one image. Exact synchronization uses a bounded 20-message queue by default.
  Data older than `maximum_data_age` is discarded, with no refreshed timestamps.
- Reliable QoS is the default for this local pipeline after best-effort large-image
  delivery dropped frames in testing. Queue bounds and age checks still apply.
  For an external best-effort camera, set `reliable_transport: false` on segmentation
  and cloud extraction or change the source QoS; reliable subscribers cannot match
  best-effort publishers. RViz's best-effort subscriptions match either publisher.
- Worker startup and stalled capture have deadlines; a failed worker stops the
  launch group. A process-level timeout cannot recover a hung kernel/driver.

## External camera input

Start with `camera:=false` to use another aligned RGB-D source. Remap the three
camera topics in the launch file or run the two processing executables with normal
ROS remapping arguments. Supply aligned depth, rectified color, and matching
rectified CameraInfo, all in the same optical frame. K must describe the supplied
pixels after any resize/ROI/binning; nonzero distortion and unadjusted ROI/binning
are rejected. CameraInfo must be published for the synchronized timestamps, not
only once at startup. For approximate synchronization, cloud timestamps follow
the depth measurement; use a small, justified slop to limit motion mismatch.

## RViz acceptance

The included configuration sets the fixed frame to `camera_color_optical_frame`
and displays RGB, mask, and `/object/cloud` with **Color Transformer: RGB8**.
For an already open RViz session, choose RGB8 under the cloud display or reload
`phase1.rviz`. The default **Move Camera** tool provides interactive perspective:
left-drag to orbit, middle-drag to pan, and scroll/right-drag to zoom. **Focus Camera**
centers the view on a clicked cloud point. The **Views** panel exposes Orbit settings.
Confirm:

1. The mask follows only the intended emergency stop.
2. The cloud contains the object's visible depth surface, with colors matching RGB.
3. Z agrees with the measured camera distance in metres; X is right and Y is down.
4. Occlusion or removal produces an empty cloud and a diagnostic reason.
5. With multiple same-class objects, association behavior is acceptable for the scene.

There is no `camera → object` TF or reference overlay yet; those belong to Phase 2.
With no object detection, an empty RViz cloud is expected. Do not treat merely
receiving cloud messages as a successful target-cloud acceptance check.

## Verification and measured performance

```bash
./scripts/test_phase1.sh -q
.venv/bin/python -m unittest discover -s tests -v
```

30 tests cover unit conversion, optical axes, masks, invalid depths/calibration,
voxel centroids, outliers, transform direction, instance association, RGB/BGR
conversion, packet bounds, actual ROS message synchronization, empty outputs,
missing TF, stale observations, per-pixel RGB correspondence, color averaging,
and packed RGB serialization. The existing suite (32 tests including the
new camera timestamp regression) passes. Package build and checkpoint CPU loading
also pass.

To measure a running pipeline in a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
.venv-ros/bin/python scripts/measure_phase1.py --seconds 15
```

Use the same `ROS_DOMAIN_ID` as the launch if overriding it. `--output path.json`
saves metrics. `--best-effort` observes an external best-effort pipeline. The probe
waits at most 35 seconds for initial data and measures for 15 seconds by default.
It subscribes to full RGB/depth images and adds some transport load.

The bounded 2026-09-23 L515 run used 960×540 RGB/aligned depth, CPU inference,
640 inference image size, two inference threads, and the actual `best.pt`:

| Measurement | Result |
| --- | --- |
| Measurement duration | 15.04 s |
| RGB / depth / mask messages | 151 / 151 / 151 |
| Cloud messages | 150 (~9.97 Hz) |
| Nonempty target clouds | 0 (no target detected) |
| Matched mask/depth timestamps | 151 |
| Matched cloud/depth timestamps | 150 |
| Segmentation median / p95 | 61.3 / 147.6 ms |
| Receipt-to-cloud median / p95 | 174.3 / 224.7 ms |

These initial measurements predate RGB fields on the cloud.
Raw results: [phase1_live_20260923.json](validation/phase1_live_20260923.json).
This confirms live transport, segmentation execution, synchronization, and no-object
behavior, **not** a validated target-cloud rate or completion of RViz acceptance.
Point preprocessing cost for a real target remains unmeasured. The target 100 ms
latency is not met; pipelined throughput near 10 Hz is a different measurement.
Timing includes acquisition receipt, camera processing/transport, sampling delay,
inference, synchronization, and cloud processing. Profile these stages with a real
object before changing resolution, image size, or CPU thread counts.

The next checkpoint is the live target/RViz check above. Then Phase 2 can load the
fixed reference once and implement centroid-seeded ICP with transform tests.

## Files and architecture

- `ros2_ws/src/object_pose_tracking/`: ROS package, three node entrypoints,
  independent geometry/segmentation modules, isolated worker/protocol, launch,
  parameter YAML, RViz configuration, and tests.
- `camera.py`: adds an optional receipt timestamp to RGBD; scanner/editor behavior
  and the existing depth/alignment contract are preserved.
- `requirements-ros.txt` and `scripts/{build,run,test}_phase1.sh`: isolated setup,
  build, launch, and deterministic checks.
- `scripts/measure_phase1.py`: bounded live throughput/latency observer.

Compatibility references: [ROS2 Python environments](https://docs.ros.org/en/jazzy/How-To-Guides/Using-Python-Packages.html)
and [Ultralytics prediction inputs and masks](https://docs.ultralytics.com/modes/predict/).
