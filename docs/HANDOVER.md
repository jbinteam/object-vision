# Project handover and hands-on guide

Prepared 2026-10-21 (Asia/Bangkok), following the myCobot280pi handover structure.
This transfers a research implementation and its operating knowledge. Physical
tracking accuracy and fresh-scene segmentation acceptance remain unfinished.
The monorepo was published on `main` at `fe232f8`; statements in imported component
guides that their roots had no Git history describe the original workstation,
not this repository. Use this checkout for future shared source changes.

## 1. What you are taking over

Two pipelines share code, environments and model ancestry:

```text
objectSegmentation:
  L515 RGB → BiRefNet foreground + assembly-specific refinement
  → reviewed versioned masks/polygon labels → background-augmented YOLO training

objectScanner:
  tagged RGB-D views → calibrated box-center poses + reviewed foreground masks
  → paired segmentation/eight-corner labels + calibrated TSDF reference
  → segmentation → object depth cloud → PnP initialization → accepted GICP pose
```

| Location | Responsibility |
| --- | --- |
| `objectSegmentation/capture_l515.py`, `l515_camera_worker.py` | RGB-only capture and Python-environment boundary |
| `remove_background.py`, `prepare_emergency_stop.py` in segmentation | General foreground extraction and red/yellow assembly refinement |
| `update_dataset.py`, `background_augmentation.py`, `train_backgrounds.py` | Versioned dataset publication, training loader and freeze audit |
| `objectScanner/camera.py`, `orbbec_camera.py` | Camera selection, rectification, alignment, timestamps and depth units |
| `turntable_dataset.py`, `scripts/capture_turntable_dataset.py` in scanner | Tag calibration, fixed box geometry and pose labels |
| `objectScanner/scripts/collect_process/` | Ordered capture/export/reference/train/live wrappers |
| `objectScanner/scripts/build_calibrated_reference.py` | Masked RGB-D TSDF in the calibrated object frame |
| `objectScanner/ros2_ws/src/object_pose_tracking/` | ROS launch/config/tests and inference/cloud/pose state machines |
| `objectScanner/webui.py`, `webui/`, `reference_geometry.py` | Saved-cloud crop and frame editor |
| Each component's `tests/`, `docs/`, requirement files | Regression checks, evidence and installation references |

Keep `objectScanner/` and `objectSegmentation/` as siblings inside the monorepo.
Scanner mask generation and training invoke the segmentation environment; training
uses its backgrounds and starting segmentation checkpoint. Current Phase 4 does
not require Isaac ROS or CenterPose. Their pinned historical dependencies are in
[DEPENDENCIES.md](../DEPENDENCIES.md).

Read the [scanner guide](../objectScanner/docs/JUNIOR_HANDOVER.md),
[segmentation guide](../objectSegmentation/docs/JUNIOR_HANDOVER.md) and
[transfer inventory](../ASSETS.md) for detail. Git contains source, frontend assets
and selected provenance metadata; it contains no complete dataset/model bundle.

## 2. Current behavior and constraints to preserve

- Segmentation's class `0 = emergency_stop` means the red button and yellow/black
  enclosure, excluding the cable. It is an image class, not a machine stop function.
- BiRefNet predicts foreground, not semantic classes or separate instances.
  Polygons are draft annotations: touching regions can merge, disconnected parts
  can split, and holes are omitted. Review full images and masks before training.
- Preserve session-based train/validation assignments and the excluded ambiguous
  frame. Failed object labeling is not an automatic negative. Background replacement
  adds appearance variety, not new physical views or independent validation scenes.
- Scanner geometry obeys `P_camera = T_camera_object @ P_object` using homogeneous
  column vectors. JSON matrices are row-major; reference coordinates are metres.
  Camera RGB is `uint8`; aligned depth is `uint16` millimetres with matching intrinsics.
- Optical `camera_color_optical_frame` is X right, Y down, Z forward. The fixed
  object frame is `box_center`. Do not re-center its reference or apply an export
  transform twice. Camera-to-robot extrinsics require separate physical calibration.
- Eight corners have IDs 0–7 in sign order `---`, `+--`, `-+-`, `++-`, `--+`, `+-+`,
  `-++`, `+++`. They are virtual box points, not observed surface features. Keep
  pose geometric flipping disabled unless semantic corner remapping is reviewed.
- PnP initializes/recovers tracking; accepted raw GICP seeds the next fit. Rejected
  candidates do not enter the seed/filter. Stale computation must not publish a new
  accepted pose or count as an ordinary rejection. Current CUDA Phase 4 accelerates
  YOLO; its Open3D GICP remains CPU, with one enabled fine iteration.
- Current Phase 4 defaults: 20 Hz publication cap, 0.75 s age gate, 0.04–1.5 m depth
  interval. These settings do not establish output rate or accuracy at 4 cm.

The available original-workstation scanner baseline is **`my_object_20260925_r2`**:
212 labeled views, 186/26 train/validation, calibration revision 2, eight corners
and a 3,811-point reference. Restore these paths together under `objectScanner/`:

| Runtime input | Required path |
| --- | --- |
| Segmentation | `training/yolo/my_object_20260925_r2_onthefly_seg/weights/best.pt` |
| Eight-corner model | `training/yolo/my_object_20260925_r2_pose_refined_pose/weights/best.pt` |
| Reference/bundle metadata | `datasets/my_object_20260925_r2/references/my_object/` |
| Immutable PnP snapshot | `datasets/my_object_20260925_r2/training_pnp/` |

The default **Gemini dataset is incomplete**: calibration/export metadata only,
no captured labels, reference, snapshot or trained pair. Bare `run_phase4*.sh`
instead selects missing historical `my_object_fresh` assets. After restoration,
select the explicit r2 session using the component guide; do not bypass guards.

Segmentation's continuation baseline is
`runs/segment/emergency_stop_frozen_backbone/weights/best.pt`, SHA256
`1e7bfd977041597a76ab15234396313940c4e817c8d904beab9ff6a061a6bb09`.
Its latest dataset has 130 records: 79 train, 50 validation, one excluded.
Updater/live defaults still select an older model; always choose weights explicitly.
Scanner's legacy `weights/best.pt` matches segmentation's
`emergency_stop_20260923T060602_397058Z` run, not the frozen baseline or scanner r2 pair.

## 3. Setup on another Linux machine

Use Linux x86-64, Python 3.10 and 3.12, Node/npm and, for the ROS path, Jazzy.
Observed versions were Python 3.10.20/3.12.3 and Node 18.19.1. Installation below
uses the existing documented CUDA 12.8 package pins; confirm the target machine's
GPU/driver suitability. A clean-machine installation has not been accepted yet.

These are **network installation steps**, separate from the offline exercise:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/jbinteam/object-vision.git
cd object-vision
# For an existing clone without initialized dependencies:
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

From `objectSegmentation/`, reproduce the model/training environment using the
[CLI setup](../objectSegmentation/CLI_CHEATSHEET.md); omit model downloading for now:

```bash
python3.12 -m venv .venv
.venv/bin/python -E -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -E -m pip install -r requirements.txt -c requirements.lock.txt
.venv/bin/python -E -m pip install -r requirements-training.txt -c requirements.lock.txt
python3.10 -m venv .venv-camera
.venv-camera/bin/python -E -m pip install -r requirements-camera.txt
```

From `objectScanner/`, create the scanner/camera environment:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-orbbec.txt
# Optional L515 backend, in this same camera environment:
.venv/bin/python -m pip install -r requirements-realsense.txt
```

Orbbec is pinned to `pyorbbecsdk2==2.1.2` (import `pyorbbecsdk`, SDK v2.9.3);
L515 uses `pyrealsense2==2.54.2.5684`. Segmentation uses Ultralytics 8.3.253 and
torch 2.8.0; ROS inference uses its separate torch 2.11.0 environments. Do not
merge them. `-E` avoids inherited Python/ROS path variables in segmentation commands.

For ROS work, first install Jazzy and the dependencies listed in
[CPU ROS setup](../objectScanner/docs/PHASE1_RGBD_SEGMENTATION.md), including colcon,
pytest and the message/TF/image packages. Then, from `objectScanner/`:

```bash
/usr/bin/python3 -m venv --system-site-packages .venv-ros
.venv-ros/bin/python -m pip install \
  'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
.venv-ros/bin/python -m pip install -r requirements-ros.txt
scripts/build_phase1.sh
# Optional CUDA ROS environment/build, separate from CPU:
python3.12 -m venv --system-site-packages .venv-ros-cuda
.venv-ros-cuda/bin/python -m pip install -r requirements-ros-cuda.txt
scripts/build_cuda.sh
```

Build scripts source Jazzy and build only the local package; they do not start
the camera. Follow [Gemini setup](../objectScanner/docs/GEMINI305_CAMERA.md) for
device permissions later. Venvs/colcon outputs contain absolute paths: recreate
them rather than assuming copied workstation environments will run.

## 4. Hands-on exercise: first day, no camera

**Fresh source-only clone:** after installation, complete these without asset restoration.

1. Open `objectScanner/docs/visualizations/object-scanner/index-light.html` in a
   browser. Step through the two streams, switch flat/isometric views and scrub
   PnP, GICP and rejected/stale states. Explain where labels come from and why the
   reference is fixed. These animations illustrate processing, not recorded telemetry.
2. From `objectScanner/`, validate the lesson source links:

   ```bash
   node docs/visualizations/object-scanner/validate-content.mjs
   ```

   Expected: `12 lessons, 53 valid source references`. Three small metadata files
   are included for these citations; their directories are not restored datasets.
3. Run synthetic/mock scanner checks from `objectScanner/`:

   ```bash
   .venv/bin/python -B -m unittest discover -s tests -p test_camera_backends.py -v
   .venv/bin/python -B -m unittest discover -s tests -p test_turntable_dataset.py -v
   .venv/bin/python -B -m unittest discover -s tests -p test_generate_yolo_seg_pose_dataset.py -v
   ```

   Expected: all selected tests pass without camera enumeration/capture. Explain
   a millimetre conversion assertion and a box-center pose-label assertion.
4. From `objectSegmentation/`, run foreground-polygon and simulated-capture checks:

   ```bash
   .venv/bin/python -B -E -m unittest discover -s tests -p test_polygons.py -v
   .venv/bin/python -B -E -m unittest discover -s tests -p test_capture_l515.py -v
   ```

   Expected: four polygon tests and two mocked capture tests pass; no weights needed.
5. To work on the crop editor, install frontend dependencies from `objectScanner/webui/`
   with `npm ci` during the network setup stage, then run `npm test` offline.
   Expected: eight geometry tests pass. The checked-in `dist/` assets already
   support the editor; run `npm run build` after changing frontend source.

**After the separate asset transfer:** follow [ASSETS.md](../ASSETS.md), preserve
relative image symlinks and audit old `/home/jb/...` paths. The frame verifier checks
absolute provenance paths, so relocation requires a reviewed metadata migration.
Do not change hashed PnP snapshots just to silence an error.

An unchanged copy restored into this nested checkout will fail the reference's
absolute-path guard even when all original hashes match. That migration is not
implemented here: archive originals, record the blocker and complete source-only
onboarding first. A later reviewed migration needs its own provenance/checksum record.

Verify the unchanged restored copy before making any relocation metadata changes:

```bash
# From each restored component directory, with all selected files present:
sha256sum -c docs/HANDOVER_ARTIFACTS.sha256
```

After the reviewed relocation, run the following from `objectScanner/` and retain
the migrated copy's own provenance/checksum record:

```bash
.venv/bin/python scripts/pnp_snapshot.py verify datasets/my_object_20260925_r2
.venv-ros-cuda/bin/python scripts/verify_phase4_frame_contract.py \
  datasets/my_object_20260925_r2 my_object \
  --calibration datasets/my_object_20260925_r2/training_pnp/calibration.json \
  --keypoints datasets/my_object_20260925_r2/training_pnp/object_keypoints.yaml
.venv/bin/python scripts/train_yolo_seg_pose.py \
  --dataset datasets/my_object_20260925_r2/yolo_seg_pose_onthefly --task both --dry-run
```

After assets and reviewed relocation: valid snapshot, eight corners and 3,811 reference
points in `box_center`, paired counts 186/26. Original hashes verify the archived copy.
A fresh source-only clone should fail asset checks with missing files; that is not
evidence of a source defect.
Inspect scanner `centerpose_masks/review/` and the saved reference (optionally
`viewer.py` on its PLY in a desktop session). Inspect segmentation's
`output/l515_dataset/latest/{validation.json,review/,previews/}` and excluded frame.
Explain cable boundaries, mask defects, split policy and axes to the mentor.

First-day done means: explain both pipelines/contracts, reproduce source-only
checks, and record environment versions/results. Asset onboarding is separately
done when checksums, path review, geometry checks and manual mask review are recorded.
No camera launch, model inference or training is needed for these exercises.

## 5. Verification and evidence

[VALIDATION.md](../VALIDATION.md) records publication checks against this packaged
source using the original environments: 121 Python AST parses; 62 scanner tests;
four segmentation polygon/two mock capture tests; 12 lessons/53 citations; and
eight frontend tests using a temporary resolver for the original pinned Three.
Ordinary `npm ci`/`npm test` on a new clone still needs to be reproduced locally.

Earlier same-day checks on complete original workspaces passed all 37 selected
artifact hashes, scanner snapshot/frame checks and the 186/26 paired-data dry run.
The [October 7 performance review](../objectScanner/docs/PHASE4_PERFORMANCE.md)
records larger ROS suites and saved-input timing comparisons. Those are dated
evidence, not new Gemini live accuracy or throughput measurements.
Save command, revision, environment, input/model hashes and output with every new
acceptance result. A clean restoration, complete regression run, physical camera
acceptance, fresh-scene model evaluation and sustained tracking remain unverified.

## 6. Ordered backlog for the new owner

| Priority | Work | Acceptance evidence |
| --- | --- | --- |
| 1 | Reproduce clean source setup | Recorded revision/submodule pins, environment install log, passing first-day checks |
| 2 | Transfer and relocate assets | Original archive manifests pass; migrated-copy provenance/checksums recorded; symlinks valid; r2 snapshot/frame checks pass |
| 3 | Review current labels and define evaluation | Full-frame/mask review, retained exclusion, independent session/negative-image plan and agreed metrics |
| 4 | Qualify segmentation on fresh scenes | Exact checkpoint/threshold; human-reviewed positives/negatives; misses, false positives and boundary errors |
| 5 | Qualify Gemini pose/frame behavior | Physical axis/metric checks, timestamps, latency, tracking losses/recovery and sustained-rate report |
| 6 | Decide new Gemini collection/retraining | New session with fixed tag/object setup; reviewed masks/split; reference/snapshot/model identities and held-out results |
| 7 | Integrate robot consumer | Measured camera-to-base extrinsics, explicit scanner root and timestamp/TRACKING agreement with robot owner |

Arrange camera work with the mentor and choose acceptance limits before collecting
results. The 4 cm software cutoff and earlier GPU benchmarks are not physical
acceptance. Preserve the r2 baseline and immutable dataset versions; choose a new
session when geometry or object placement changes. Retraining is not an installation
test. Use explicit checkpoints, fresh run names and retain args/results/freeze audits.

## 7. Troubleshooting and project boundaries

| Symptom | First check |
| --- | --- |
| Missing weights/data in fresh clone | Separate asset transfer; provenance copies do not contain payloads |
| Snapshot hash or frame guard fails | Exact paired calibration/reference/model set; reviewed relocation paths; do not bypass validation |
| Default live wrapper reports missing files | Incomplete Gemini session; select restored r2 explicitly |
| Bare Phase 4 cannot find `my_object_fresh` | Historical launch defaults; follow the explicit-session component workflow |
| SDK/ROS import error | Camera Python 3.10 vs ROS Python 3.12, correct venv and sourced Jazzy; segmentation `-E` |
| `three` missing during frontend test | Run `npm ci` inside scanner `webui/` during installation |
| Mask includes wire/background or labels merge | Full-frame review, foreground/refinement assumptions and polygon limitations |
| False positives or apparently good validation only | Exact selected checkpoint and independent scenes; synthetic backgrounds do not establish generalization |

`download_model.py --help` actually downloads: it has no argument parser.
Camera `--list` accesses hardware; `rviz:=false` does not make a camera launch
offline. Updater `--dry-run` can create a directory/lock. Keep these outside the
source-only exercise. Optional CenterPose/Isaac experiments have separate setup
and licenses; their results do not qualify the active YOLO/PnP/GICP path.

The public object-pose illustration is hosted by the separate `nn_ik/webui` project;
its deployed state was not rechecked here. Robot control belongs to myCobot280pi:
its full launcher must override `--scanner-root` to this checkout's
`objectScanner/`, since the monorepo nesting differs from old sibling defaults.
Robot launch/arming is outside this onboarding. Save new shared work in this Git
repository and keep experimental data/backups separately as described in ASSETS.md.
