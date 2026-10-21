# objectSegmentation: junior engineer handover

Prepared 2026-10-21 from local files and lightweight offline checks.
Start with this guide, then [README](../README.md) and [CLI cheat sheet](../CLI_CHEATSHEET.md).
The cross-project transfer entry point is [PROJECT_HANDOVER.md](../../objectScanner/PROJECT_HANDOVER.md).

## Purpose and current delivery

This project turns RealSense L515 color captures into foreground masks, cutouts,
and draft single-class YOLO segmentation labels, then trains/tests YOLO11 models.
The implemented application is **class 0 = emergency_stop**: the red button and
yellow/black enclosure, excluding its cable. This is a vision class name;
the project does not implement a machine emergency-stop control function.

The local workflow and trained artifacts are available for continuation.
Model accuracy in new scenes remains an open acceptance task. Camera capture,
training, downloads, and deployment were not run during this handover.

```text
L515 RGB -> input/l515/<session>/
  -> BiRefNet + object-specific refinement -> versioned images/masks/YOLO labels
  -> reviewed train/validation split -> random-background YOLO training
  -> selected checkpoint -> live RGB segmentation and saved predictions
```

## Code map

| File | Responsibility / things to preserve |
|---|---|
| `capture_l515.py` | RGB PNG capture, camera/session metadata; no depth recording |
| `remove_background.py` | General BiRefNet foreground extraction and optional class-0 polygons |
| `prepare_emergency_stop.py` | Red/yellow object selection, crop refinement and cable removal |
| `update_dataset.py` | Incremental labeling, persistent session splits, validation and atomic publication |
| `background_augmentation.py` | Custom Ultralytics loader; replace training backgrounds using masks |
| `train_backgrounds.py` | Preview/train entry point and frozen-parameter/BatchNorm audit |
| `test_l515.py` | Model inference/preview, snapshots and prediction metadata |
| `l515_camera_worker.py`, `l515_protocol.py` | Camera subprocess and frame transport across Python environments |
| `download_model.py`, `download_backgrounds.py` | Explicit network setup utilities |
| `tests/` | Polygon, simulated camera, updater, augmentation and freezing checks |
| `BiRefNet/` | Clean nested upstream Git checkout; separate from project wrappers |

The project root itself has **no Git repository**. Preserve a full file backup,
including wrappers, requirements, docs and hidden project configuration.
`git -C BiRefNet status` describes upstream code only.

## Pinned environments and model assets

Use `python -E` as documented to ignore inherited Python/ROS path variables.

| Component | Verified local setup |
|---|---|
| Model / training Python | `.venv`, Python 3.12.3 |
| GPU libraries | torch 2.8.0+cu128, torchvision 0.23.0+cu128 |
| Trainer | ultralytics 8.3.253; custom dataset depends on this API |
| Image libraries | numpy 1.26.4, opencv-python 4.11.0.86 |
| Camera Python | `.venv-camera`, Python 3.10.18 |
| Camera packages | pyrealsense2 2.54.2.5684; separate `requirements-camera.txt` |
| BiRefNet source | `ebcc0bc8ec7fe919cec829f2dea656b3078acddc` |
| BiRefNet model ID/revision | `ZhengPeng7/BiRefNet` / `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` |

The pinned model revision is declared in `download_model.py`; the local files alone
do not independently prove their download origin. Keep `models/birefnet/config.json`
and `model.safetensors` together. They are regular files, about 424 MiB total.
The wrapper loads the local upstream implementation with `bb_pretrained=False`.
No external Hugging Face cache dependency was found in the inspected pipeline;
a clean-machine cold model load was not tested. Retain the local weight files.

For a replacement machine, use README's **Reproduce the environment** section and
CLI cheat sheet's **Downloads, environment setup, and checks**. Install the pinned
CUDA PyTorch pair/index first, then `requirements.txt` and
`requirements-training.txt` constrained by `requirements.lock.txt`.
Recreate the camera environment separately with Python 3.10 (documented wheels also
target 3.11). Existing virtual environments contain machine-specific paths;
their transfer is a convenience snapshot, not a portable installation guarantee.

The current camera environment references an interpreter under
`/home/jb/anaconda3/envs/cv_py310/`; copying `.venv-camera` alone may break it.
The recorded developer GPU was an RTX 4070 Laptop GPU with 8 GB VRAM.
Hardware/driver compatibility on the junior's machine must be checked there.

## Dataset to transfer and inspect first

`output/l515_dataset/latest` currently resolves to:
`versions/20260923T060602_397058Z`.
Its `validation.json` and split text files agree on:

| Records | Training | Validation | Excluded |
|---|---|---|---|
| 130 labeled | 79 | 50 | 1 |

The excluded key is first-session `frame_000041_20260923T045728_559446Z.png`
under `20260923T045549_326628Z_924e6242`: an additional partly visible emergency-stop
candidate is outside the primary mask. Retain its exclusion and review flag.

Inspect `latest/manifest.json`, `validation.json`, `review/` and `previews/`.
The manifest records source/pixel checksums, session assignments, origin and flags.
The report's polygon/mask IoU minimum is 0.99503; that checks representation
agreement, not agreement with human ground truth.
Published dataset versions and source captures should remain immutable.
Keep `input/l515/`, the original base `output/l515_emergency_stop/`,
`output/l515_100/`, all `output/l515_dataset/` versions/caches and the `latest` link.
The updater depends on these assets for provenance and incremental reuse.

`data.yaml` embeds an absolute dataset root; split text files use relative
`./images/...` entries. After relocation, preserve the archive and create an
adapted dataset copy/config with the new root before training. Manifest/history
paths also refer to the old workspace; audit updater resume behavior before use.

## Checkpoint selection and evidence

For the next deliberate segmentation experiment, start from the documented
frozen-backbone checkpoint and always pass its path explicitly:
`runs/segment/emergency_stop_frozen_backbone/weights/best.pt`.
It is the documented continuation baseline, not a claim that it is the best
model for unseen scenes. Retain its full run directory and the earlier runs.

Evidence in `args.yaml`, `results.csv`, `background_augmentation.json` and
`freeze_audit.json`: the expanded 79/50 dataset; 53 logged epochs of a 100-epoch
limit; patience 25; freeze 11; AdamW; learning rate 0.0001; 85% random background
replacement. The audit reports 1,365,472 frozen parameters and 1,477,315 trainable,
with identical initial/final frozen-state hashes. README reports best epoch 28;
this agrees with combined box/mask fitness from the CSV, not mask mAP alone.

SHA256 of this checkpoint:
`1e7bfd977041597a76ab15234396313940c4e817c8d904beab9ff6a061a6bb09`.
The updater and live tester still default to the older
`emergency_stop_backgrounds/weights/best.pt`; neither selects the latest run.
Each `--train` starts fresh optimizer state, even when no images changed.

## Relationship to objectScanner

Keep sibling layout: `projects/objectScanner/` and `projects/objectSegmentation/`.
Scanner's `scripts/augment_centerpose_backgrounds.py` uses this project's BiRefNet
code/weights, and scanner's collection/training wrappers invoke this `.venv`.
Scanner's `scripts/train_yolo_seg_pose.py` defaults to this frozen checkpoint and
`backgrounds/picsum/`. Its derived segmentation/8-keypoint pose datasets and
trained models belong to scanner and must be transferred separately.

Scanner's runtime `weights/best.pt` is byte-identical to this project's
`runs/segment/emergency_stop_20260923T060602_397058Z/weights/best.pt` (SHA256
`83a9dca7ade6a95f1811bd711424c49c449c1f9c6b9262eae5b53eebc821feb0`).
It differs from the frozen checkpoint above. Byte equality establishes the
shared artifact, but does not establish the historical copy command or approval
to replace scanner's runtime model. Preserve scanner's pose/calibration contracts.

## Known limitations and first improvement target

Labels are automatic drafts. BiRefNet finds foreground rather than semantic
classes; touching objects may merge, disconnected parts split, and polygon holes
are omitted. Red/yellow refinement is tailored to this assembly and needs those
visible cues. Failed/uncertain detections are excluded, never automatic negatives.
Do not send negative-only scenes through the automatic object labeler.

Splits are separated by session, but both original sessions contain the same
physical object in a similar setting. Thirty later views were added only to
training. The existing validation set does not independently measure those views
or generalization. Exact pixel deduplication does not detect near-duplicate frames.
Background replacement adds scene appearance variety, not new geometry/lighting.
An earlier live test recorded a filament spool false positive around confidence
0.51; see README and `output/l515_test/20260923T054318_889054Z_913d01e5/`.
That evidence concerns the then-used model, not a fresh evaluation of the frozen run.

## Offline onboarding: run individually from the project root

```bash
readlink output/l515_dataset/latest
cat output/l515_dataset/latest/validation.json
sha256sum -c docs/HANDOVER_ARTIFACTS.sha256
.venv/bin/python -B -E -m pip check
.venv/bin/python -B -E -m unittest discover -s tests -p test_polygons.py -v
.venv/bin/python -B -E -m unittest discover -s tests -p test_capture_l515.py -v
```

Handover verification: both commands passed (4 polygon + 2 mocked capture tests),
and `pip check` found no broken requirements. These checks use no physical camera,
training, downloads or inference. README's historical 31-test result was not rerun.
Safe CLI inspection: `update_dataset.py --help`; `--dry-run` avoids labeling but
may create a dataset directory/lock file. **`download_model.py --help` downloads**:
it has no argument parser. Do not use it as an onboarding help command.

## Future deliberate operations: after onboarding and task agreement

For live capture/testing, follow the CLI cheat sheet with the camera connected
and explicit checkpoint selection. Stop preview with Q/Esc or Ctrl+C.
First record a fresh validation session and negatives with human-reviewed labels;
choose `--val-session SESSION` on first import, before it defaults to training.
Existing training assignments cannot be moved to validation via that flag.
Agree on a held-out evaluation and mask/false-positive acceptance criteria before
claiming improvement. Camera `--list` is a hardware operation, not an offline test.

For training, use `update_dataset.py --train --model` with the explicit frozen
checkpoint above, or `train_backgrounds.py` for detailed settings. Plain `yolo train`
does not install the custom background loader. Use a fresh run name and retain
source checkpoint, dataset version, arguments, freeze audit and validation results.

## First-week checklist and transfer acceptance

- Day 1: receive both sibling projects; verify artifact hashes, dataset link/counts,
  paths and environment availability. Read both handover guides and CLI references.
- Day 2: pass offline checks; inspect full-frame previews and the excluded frame;
  explain class boundary/cable policy and the scanner dependencies to the senior.
- Day 3: inventory new-scene failure cases; propose independent capture sessions,
  human negative labels and measurable evaluation before changing the model.
- Day 4: after agreement, run a bounded camera evaluation and record exact weights,
  threshold, false positives/misses and mask defects; keep predictions separate from labels.
- Day 5: review evidence with the senior, then choose the next dataset/training task.
  Record decisions, known failures and commands in `HANDOFF.md`.

Transfer raw captures, all dataset versions/caches, BiRefNet code/model, all runs,
`yolo11n-seg.pt` and `yolo11n.pt`, background photos and `sources.json`, tests,
requirements, docs and hidden project state. Current main storage is approximately
3.9 GiB output, 351 MiB input, 424 MiB models, 76 MiB runs and 11 MiB backgrounds;
environments add about 7.6 GiB. Preserve symlinks and provenance, and verify on arrival.
Do not rely on an upstream Git clone to recover this workspace's data or wrappers.
