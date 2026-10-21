# objectScanner: junior engineer handover

Prepared 2026-10-21 from the local workspace. Start with the joint
[project transfer index](../PROJECT_HANDOVER.md); this guide covers scanner operation
and maintenance. [HANDOFF.md](../HANDOFF.md) is session continuity, not an instruction
to restart old jobs. This directory has no `.git` repository or recoverable Git HEAD.

## What you are taking over

The current engineering path is calibrated RGB-D collection → reviewed masks →
paired YOLO segmentation/eight-corner training → fixed reference → ROS2 Phase 4
PnP/GICP tracking. The known object is `my_object`; output is its six-dimensional
pose relative to the camera. Robot-base output additionally needs an independently
calibrated camera-to-base TF. Robot manipulation is not accepted by this handover.

The working asset set is **`my_object_20260925_r2`**, originally collected with L515:
212 labels, 186 training views, 26 validation views, two trained checkpoints,
calibration revision 2, frozen PnP inputs and a cropped 3,811-point TSDF reference.
Gemini 305 is now the default acquisition backend. Its session
`datasets/my_object_gemini305/` currently has calibration/export metadata and
calibration history only: **zero captured labels, no production reference,
no training snapshot and no session checkpoints**. It is not ready for step 7.

Use an explicit session throughout. Bare `run_phase4*.sh` still selects historical
`my_object_fresh` assets that are absent here. Bare collection wrappers select the
unfinished Gemini session. Neither default identifies the available r2 asset set.

Earlier scanner/Phase 1–3, CUDA ICP, CenterPose, NOCS and diagnostic scan workflows
remain in the tree. They explain development history, but are not interchangeable
with Phase 4. In particular, Phase 3 CUDA ICP performance does not describe Phase 4
GICP. The old [tracking plan](OBJECT_POSE_TRACKING_PLAN.md) predates implemented
Phase 4; use current code and [Phase 4](PHASE4_YOLO_PNP_GICP.md) for present behavior.

## Read and edit map

| Area | Main files and responsibility |
| --- | --- |
| Acquisition | `camera.py`, `orbbec_camera.py`: backend choice, alignment, intrinsics and depth normalization |
| General scanning | `scanner.py`, `reconstruction.py`, `supervisor.py`: moving-camera fusion and watchdog; `viewer.py` inspects clouds |
| Calibrated collection | `scripts/capture_turntable_dataset.py`, `turntable_dataset.py`: tags, box calibration and labels |
| Collection orchestration | `scripts/collect_process/01_capture.sh` through `07_run_live.sh`; shared defaults in `common.sh` |
| Reference generation | `scripts/build_calibrated_reference.py`; `reference_geometry.py`, `webui.py`, `webui/src/` handle crop/frame editing |
| Export/training | `scripts/generate_yolo_seg_pose_dataset.py`, `scripts/train_yolo_seg_pose.py`, `yolo_background_augmentation.py` |
| Geometry integrity | `scripts/pnp_snapshot.py`, `scripts/verify_phase4_frame_contract.py` |
| ROS package | `ros2_ws/src/object_pose_tracking/`; launch files, YAML presets and tests live together |
| ROS acquisition | Package `camera_node.py`, `camera_worker.py`, `protocol.py`: Python 3.10 SDK worker → Python 3.12 ROS |
| ROS inference/cloud | `observation_node.py`, `segmentation.py`, `cloud_node.py`, `geometry.py` |
| ROS pose/state | `pnp_gicp_node.py`, `pnp.py`, `gicp.py`, `tracking.py`, `pose_filter.py`, `pose_outputs.py` |
| Tests/evidence | Root `tests/`, package `test/`, dated `docs/validation/` reports |
| Teaching/site assets | `docs/visualizations/object-scanner/`; its README explains rebuilding/exporting |

Package filenames in the last three ROS rows are relative to
`ros2_ws/src/object_pose_tracking/object_pose_tracking/`.

## Environments and dependencies

Keep the Python environments separate; use explicit interpreter paths.
The versions below were checked from local environment metadata on 2026-10-21.

| Environment | Purpose and pinned dependencies |
| --- | --- |
| `.venv` | Python 3.10.20; camera/scanner/editor; NumPy 1.26.4, Open3D 0.19.0, OpenCV 4.10.0.84, pupil-apriltags 1.0.4.post11, Flask 3.1.3 |
| `.venv-ros` | Python 3.12.3 with system site packages; ROS2 Jazzy CPU environment; pins in `requirements-ros.txt` |
| `.venv-ros-cuda` | Python 3.12.3 with system site packages; torch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.3.253, Open3D 0.19.0, Triton 3.6.0 |
| `../objectSegmentation/.venv` | Scanner YOLO training interpreter and shared background library; training API pinned to Ultralytics 8.3.253 |
| `.venv-centerpose` | Historical Python 3.12.3 CenterPose experiments; separate `requirements-centerpose.txt` and patch |

Camera SDK pins are `pyorbbecsdk2==2.1.2` (import `pyorbbecsdk`, embedded SDK
v2.9.3) and optional `pyrealsense2==2.54.2.5684`. Both are installed in `.venv`.
Do not combine the camera SDK worker with ROS Python to simplify installation.
Recreate rather than relocate venvs; they and colcon entrypoints contain absolute paths.

Installation/build references: [README installation](../README.md),
[CPU ROS setup](PHASE1_RGBD_SEGMENTATION.md), [CUDA setup](CUDA_TRACKING.md),
[Gemini setup/udev](GEMINI305_CAMERA.md), and the sibling project's handover.
On a replacement host, first reproduce Python 3.10/3.12 and `/opt/ros/jazzy`,
install the corresponding requirements, then build using `scripts/build_phase1.sh`
and/or `scripts/build_cuda.sh`. Those builds do not start a camera.
The reference-editor frontend has included compiled assets; use `npm ci`,
`npm run build`, `npm test` inside `webui/` when changing it.

## Preserve this runtime set together

Paths below are relative to the project root and were present during this handover.
The joint index supplies checksums; the filenames alone are not identity checks.

| Artifact | Exact path |
| --- | --- |
| Segmentation | `training/yolo/my_object_20260925_r2_onthefly_seg/weights/best.pt` |
| Pose | `training/yolo/my_object_20260925_r2_pose_refined_pose/weights/best.pt` |
| Production reference | `datasets/my_object_20260925_r2/references/my_object/reference.ply` |
| Reference metadata | Same directory: `frame.json`, `object_keypoints.yaml`, `reference_bundle.zip` |
| Frozen PnP inputs | `datasets/my_object_20260925_r2/training_pnp/{calibration.json,object_keypoints.yaml,manifest.json}` |
| Runtime parameters | `ros2_ws/src/object_pose_tracking/config/phase4.yaml`, `devices_cuda.yaml` |

There is no `reference_aligned.ply` in this calibrated reference: the production
PLY already occupies `box_center`. The live wrapper verifies the snapshot hashes
and reference/keypoint contract before ROS starts; do not bypass a failed guard.

Cross-project provenance is verified: scanner `weights/best.pt` is byte-identical
to sibling `runs/segment/emergency_stop_20260923T060602_397058Z/weights/best.pt`.
It is the legacy Phase 1 `emergency_stop` model, not the current `my_object` model.
The r2 segmentation run's `args.yaml` names the sibling
`runs/segment/emergency_stop_frozen_backbone/weights/best.pt` as its starting model;
it trained on scanner r2 labels with `freeze: 11`. The r2 pose run starts from
scanner `training/yolo/my_object_rgbd_unfrozen_pose/weights/best.pt`, with
`freeze: 0` and `lr0: 0.00005`. Preserve those source runs for retraining provenance.

## Frame, unit and tracking contracts

- `P_camera = T_camera_object @ P_object`, with homogeneous column vectors;
  stored matrices are row-major arrays. Exported references and geometry use metres.
- Camera output is aligned, rectified RGB `uint8` and depth `uint16` millimetres,
  with output-image intrinsics. ROS encodings are `rgb8`/`16UC1`; the optical frame
  is `camera_color_optical_frame`: X right, Y down, Z forward, zero distortion.
- `box_center` is the fixed calibrated object origin. Do not re-center the PLY,
  invert `T_camera_object`, or apply the editor export transform again at runtime.
- Eight corners are ordered by signs: `---`, `+--`, `-+-`, `++-`, `--+`, `+-+`,
  `-++`, `+++` for IDs 0–7. r2 dimensions are approximately
  `0.100351925 × 0.094271602 × 0.092145361 m`. Read exact snapshot coordinates.
- Corners are virtual box points; a label's visibility 2 means finite projection
  inside the image, not physical surface visibility. Keep geometric flips disabled
  (`fliplr=0`) unless a reviewed semantic remapping is implemented.
- Segmentation runs during tracking; pose keypoints/PnP initialize or recover it.
  GICP uses the last accepted **raw** pose as the next seed. Rejected candidates
  never replace that seed or enter the temporal filter. Stale computation returns
  without publishing a new accepted pose or counting as an ordinary rejection.
- Current Phase 4 uses CPU Open3D GICP even with CUDA YOLO; coarse/medium are
  disabled and fine has one iteration. Read actual YAML rather than old comments.
- Defaults are 20 Hz publication cap, 0.75 s data-age gate and 0.04–1.5 m depth
  cutoffs. A cutoff of 4 cm is not measured accuracy or tracking acceptance there.
  Gemini replacement requires a new camera-to-robot calibration.

## Offline onboarding: no camera, GPU inference or training

Run from `/home/jb/codex/projects/objectScanner` with the preserved r2 dataset:

```bash
.venv/bin/python scripts/pnp_snapshot.py verify datasets/my_object_20260925_r2
.venv-ros-cuda/bin/python scripts/verify_phase4_frame_contract.py \
  datasets/my_object_20260925_r2 my_object \
  --calibration datasets/my_object_20260925_r2/training_pnp/calibration.json \
  --keypoints datasets/my_object_20260925_r2/training_pnp/object_keypoints.yaml
.venv/bin/python scripts/train_yolo_seg_pose.py \
  --dataset datasets/my_object_20260925_r2/yolo_seg_pose_onthefly --task both --dry-run
```

Checked now: snapshot verification passed; frame verification reported eight corners
and 3,811 reference points in `box_center`; dry-run paired counts were 186/26 for
both tasks. Dry-run validates pairs and prints a plan; it does not prove model
quality or availability of every starting checkpoint. No full test suite was rerun
for this documentation handover. For later changes, select focused root/package
tests first; `scripts/test_phase1.sh` and `scripts/test_cuda.sh` are broader suites.

## Deliberate operations for a later supervised session

The following commands are operational examples, not steps performed at handover.
Close other camera users before a live launch. The wrapper opens the camera;
`rviz:=false` only disables the viewer. Ctrl+C stops the ROS launch group.

```bash
# Existing validated-on-disk r2 assets, with current Gemini backend:
scripts/collect_process/07_run_live.sh my_object_20260925_r2 my_object rviz:=false
# New capture: choose a fresh session; replace SERIAL with the selected camera.
scripts/collect_process/01_capture.sh junior_object_gemini305 my_object --serial SERIAL
```

For new data, follow [collection instructions](../scripts/collect_process/README.md)
one step at a time: capture → masks/review → paired export → calibrated reference
→ segmentation training → pose training → live. Steps 5/6 start GPU training and
write snapshots/runs; they depend on the sibling environment/backgrounds.
Do not run them merely to check installation or overwrite the baseline session.
Keep the object fixed relative to permanent tags; current setup uses 14 mm temporary
IDs 1–4 and 28 mm permanent IDs 5–10 in `tagStandard41h12`, with at least two mapped
permanent tags visible per captured view. Optional `P` scanning uses CUDA and is
diagnostic; step 4's masked calibrated RGB-D TSDF is the production reference.

## Open acceptance and the junior's first week

Recorded reports in [performance review](PHASE4_PERFORMANCE.md) include 159 CPU
ROS tests (15 optional skips), 104 focused CUDA tests, builds and saved-input timings
from 2026-10-07. [Gemini notes](GEMINI305_CAMERA.md) record firmware 1.0.70 streaming
validation that day. Those are historical evidence, not checks repeated today.
The camera note's old 30 Hz statement conflicts with the current 20 Hz launch default.

Still open: Gemini end-to-end latency, moving-object pose/frame acceptance,
sustained tracking rate, long-duration GPU stability and metric depth/recognition
accuracy near 4 cm. Robot-base output needs physical extrinsic acceptance.
Do not present prerecorded-input speedups or teaching animations as live accuracy.

1. Day 1: copy both project trees and checksums; verify r2 assets and environment
   identities; read this guide and inspect saved RGB/masks/reference without capture.
2. Day 2: trace camera → observation → cloud → pose modules and frame contracts;
   run focused PnP/tracking tests in the documented environment, without live nodes.
3. Day 3: review mask quality/split policy, run training dry-run, inspect training
   `args.yaml`/results and source checkpoints. Agree measurable live acceptance targets.
4. Day 4: with the supervisor, confirm camera/firmware/USB and physical alignment;
   explicitly launch the r2 assets and record latency, losses/recovery and overlay error.
5. Day 5: document results and defects; decide whether a fresh Gemini collection is
   needed. Create a separate session for changed geometry or object placement.

## Transfer and recovery

Keep source/docs/tests, full `datasets/my_object_20260925_r2/`, Gemini calibration
history, relevant `training/` runs and their args/results/logs, `weights/`,
`references/`, camera configuration and sibling backgrounds/source checkpoints.
Dataset exports contain relative image symlinks: copy their targets and preserve
links. `frame.json`, manifests and training YAML also contain absolute paths;
the frame verifier checks provenance paths. Prefer the original sibling layout
initially; a changed root needs a reviewed path migration and verification, not
manual changes to frozen PnP snapshots. Retain hidden pre-crop reference backups.

Back up the current tree before cleanup; there is no Git rollback. Build/install
trees and venvs can be recreated, but are useful environment evidence. Historical
`output/` logs, scans, validation/deployment evidence and the animation source should
be retained in an archive. The separate Arm Lab host/site is described in session
history; publishing or changing it is outside this project transfer. See the joint
index for the cross-project inventory, backup priorities and checksum verification.
