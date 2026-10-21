# Project CLI cheat sheet

Commands and defaults checked against the local scripts on 2026-09-23.
Run commands individually from the project root:

```bash
cd /home/jb/codex/projects/objectSegmentation
```

Use `.venv-camera/bin/python -E` for camera-only capture and
`.venv/bin/python -E` for labeling, training, and live inference. The live tester
starts the camera environment automatically. `-E` ignores inherited Python paths.
Replace uppercase placeholders such as `SESSION_FOLDER` and `RUN_NAME` before use.
Choose a fresh output name each time an example requires an empty directory.

## Everyday workflow: capture → update → retrain → test

```bash
# 1. Capture 30 new images, two seconds apart (camera required).
.venv-camera/bin/python -E capture_l515.py --auto --interval 2 --count 30

# 2. Inspect pending dataset work without labeling or publishing.
.venv/bin/python -E update_dataset.py --dry-run

# 3. Update the dataset and fine-tune from the existing frozen-backbone checkpoint.
.venv/bin/python -E update_dataset.py --train \
  --model runs/segment/emergency_stop_frozen_backbone/weights/best.pt \
  --freeze 11 --optimizer AdamW --lr0 0.0001 --epochs 100 --batch 4

# 4. Use the NEW run's name printed by training.
.venv/bin/python -E test_l515.py \
  --weights runs/segment/RUN_NAME/weights/best.pt
```

New runs go under `runs/segment/`; existing runs and checkpoints are preserved.
The updater uses `emergency_stop_<dataset-version>` as the run name, with a suffix
when needed. The model path is explicit: it does not automatically select your
most recently trained checkpoint. Each `--train` invocation starts a new run with
fresh optimizer state, even if no images were added.

## Freeze and training defaults

| Setting | `update_dataset.py --train` | `train_backgrounds.py` directly |
|---|---|---|
| Starting checkpoint | `runs/segment/emergency_stop_backgrounds/weights/best.pt` | `yolo11n-seg.pt` |
| Frozen modules | `--freeze 11` | `--freeze 0` |
| Optimizer | `AdamW` | `auto` |
| Initial learning rate | `0.0001` | Chosen by automatic optimizer unless explicitly set |
| Epoch limit / batch | `100` / `4` | `100` / `4` |
| Device | GPU `0` | GPU `0` |
| Image size / loader workers | `640` / `2` via trainer defaults | `640` / `2` |
| Random background probability | `0.85` via trainer default | `0.85` |

`--freeze 11` freezes YOLO11 modules **0–10**, the complete backbone. The neck and
segmentation head remain trainable. `--freeze 0` trains all normally trainable
parameters; the model's fixed DFL projection still stays frozen. A smaller positive
count freezes only the first N modules. Counts depend on model architecture.

Training stops early after 25 epochs without improvement; `--epochs` is a maximum.
The wrappers do not expose `--resume` or `--patience`. Use an explicit `AdamW` or
`SGD` optimizer with `--lr0`; the direct trainer rejects a learning rate with `auto`.

## Capture L515 images

```bash
# List connected cameras and supported RGB modes.
.venv-camera/bin/python -E capture_l515.py --list

# Manual capture with preview.
.venv-camera/bin/python -E capture_l515.py

# Capture 100 images automatically.
.venv-camera/bin/python -E capture_l515.py --auto --interval 2 --count 100

# Capture without a preview window.
.venv-camera/bin/python -E capture_l515.py --auto --interval 2 --count 30 --no-preview

# Select a camera and mode reported by --list.
.venv-camera/bin/python -E capture_l515.py \
  --serial CAMERA_SERIAL --width 1280 --height 720 --fps 30
```

Controls: **Space/S** saves; **R** toggles timed capture; **Q/Esc** exits.
Each run creates `input/l515/<session>/` with color PNGs, `session.json`, and
`frames.jsonl`. There is a two-second exposure warmup. No depth images are saved.

## Update the emergency-stop dataset

```bash
# Label new/retry failed captures and publish a dataset; do not train.
.venv/bin/python -E update_dataset.py

# Report pending captures. This may create the dataset directory/lock file.
.venv/bin/python -E update_dataset.py --dry-run

# Reserve a NEW capture session for validation on its first import.
.venv/bin/python -E update_dataset.py --val-session NEW_SESSION_FOLDER

# Reserve multiple new validation sessions.
.venv/bin/python -E update_dataset.py \
  --val-session FIRST_NEW_SESSION --val-session SECOND_NEW_SESSION

# Use the updater's defaults, including the ORIGINAL trained checkpoint.
.venv/bin/python -E update_dataset.py --train

# Fine-tune all normally trainable layers from a chosen checkpoint.
.venv/bin/python -E update_dataset.py --train \
  --model runs/segment/RUN_NAME/weights/best.pt --freeze 0

# Update and train on CPU.
.venv/bin/python -E update_dataset.py --train --device cpu \
  --model runs/segment/emergency_stop_frozen_backbone/weights/best.pt
```

Input defaults to `input/l515/<session>/*.png`. New sessions default to training;
existing split assignments cannot be changed through `--val-session`.
The latest complete dataset is `output/l515_dataset/latest/data.yaml`.
Review `latest/review/`, `latest/previews/`, `manifest.json`, and `validation.json`.
Existing source images must retain their contents; add revised captures as a new session.

The labeler is specific to the red button/yellow enclosure and excludes cables.
Do not feed negative-only scenes through this automatic labeling workflow.
`--dry-run` and `--train` cannot be combined.

## Direct training and augmentation previews

Use the direct trainer to select a dataset, run name, image size, worker count,
or background settings. It uses the latest incremental dataset when available,
otherwise `output/l515_emergency_stop/data.yaml`.

```bash
# Train from the base pretrained segmentation model, with no user-frozen modules.
.venv/bin/python -E train_backgrounds.py \
  --model yolo11n-seg.pt --epochs 100 --batch 4 --imgsz 640 \
  --workers 2 --device 0 --name emergency_stop_new

# Fine-tune the existing checkpoint with its backbone frozen.
.venv/bin/python -E train_backgrounds.py \
  --data output/l515_dataset/latest/data.yaml \
  --model runs/segment/emergency_stop_frozen_backbone/weights/best.pt \
  --freeze 11 --optimizer AdamW --lr0 0.0001 \
  --epochs 100 --batch 4 --name emergency_stop_frozen_next

# Select background probability (1 = always replace; 0 = retain original scene).
.venv/bin/python -E train_backgrounds.py \
  --backgrounds backgrounds/picsum --background-probability 1 \
  --name emergency_stop_all_backgrounds

# Generate 12 examples through the training transforms, then exit without training.
.venv/bin/python -E train_backgrounds.py --preview 12 \
  --preview-output output/background_preview_new
```

Previews require a new/empty output directory. First four samples reuse the same
source image to show independent augmentation. Validation uses original images.
Training saves `weights/best.pt`, `weights/last.pt`, `results.csv`, plots, and
`background_augmentation.json`; freezing also produces `freeze_audit.json`.
Use this wrapper for background replacement; plain `yolo train` does not install
the project's custom loader.

## Live model testing

```bash
# Test the existing frozen-backbone checkpoint.
.venv/bin/python -E test_l515.py \
  --weights runs/segment/emergency_stop_frozen_backbone/weights/best.pt

# Default tester uses the ORIGINAL emergency_stop_backgrounds checkpoint.
.venv/bin/python -E test_l515.py

# List cameras through the camera worker.
.venv/bin/python -E test_l515.py --list

# Use a selected checkpoint and confidence threshold.
.venv/bin/python -E test_l515.py \
  --weights runs/segment/RUN_NAME/weights/best.pt --conf 0.7

# Headless test: 30 predictions, save every 15th prediction.
.venv/bin/python -E test_l515.py \
  --weights runs/segment/emergency_stop_frozen_backbone/weights/best.pt \
  --no-preview --max-frames 30 --save-every 15

# Full-resolution camera input, inference on CPU.
.venv/bin/python -E test_l515.py \
  --weights runs/segment/emergency_stop_frozen_backbone/weights/best.pt \
  --width 1920 --height 1080 --fps 30 --device cpu
```

Controls: **S/Space** saves; **M** toggles masks; **+/−** adjusts confidence;
**Q/Esc** exits. Results go into `output/l515_test/<session>/`, including original
and annotated snapshots, masks, prediction metadata, and a session summary.

## General foreground removal and initial dataset refinement

```bash
# Remove backgrounds from one image or a recursively searched directory.
.venv/bin/python -E remove_background.py /path/to/photos --output output/my_batch

# Also export single-class draft YOLO polygons (class ID 0).
.venv/bin/python -E remove_background.py input/l515/SESSION_FOLDER \
  --output output/my_object --class-name emergency_stop

# Adjust inference/mask/polygon settings; use CPU.
.venv/bin/python -E remove_background.py /path/to/photos \
  --output output/my_cpu_batch --device cpu --size 1024 \
  --threshold 0.5 --min-area 100 --epsilon 0.001 --class-name bottle

# Recreate object-specific refinement from the existing first-pass dataset.
.venv/bin/python -E prepare_emergency_stop.py \
  --raw output/l515_100 --output output/emergency_stop_new
```

Both output directories must be new/empty. Background-removal output must also
be outside the input directory. It produces `cutouts/`, `alpha/`, `masks/`,
`previews/`, and `manifest.json`; `--class-name` adds `images/` and `labels/`.
BiRefNet extracts foreground, so touching objects may merge and disconnected
parts may split. Review labels before training.

`prepare_emergency_stop.py` needs the first-pass manifest, masks, and original
source files; it is the initial dataset-specific preparation tool. Use
`update_dataset.py` for ongoing additions.

## Downloads, environment setup, and checks

The environments already exist locally. Setup commands below are for recreating
missing components; the pinned versions come from this project's setup.

```bash
# Download/cache pinned BiRefNet weights. This script accepts no CLI options.
.venv/bin/python -E download_model.py

# Download background photos; completed images are reused.
.venv/bin/python -E download_backgrounds.py --count 100

# Create a separate background collection.
.venv/bin/python -E download_backgrounds.py \
  --output backgrounds/custom --count 100 --seed 42 --width 1280 --height 720

# Check installed model-environment dependencies.
.venv/bin/python -E -m pip check

# Run project tests.
.venv/bin/python -E -m unittest discover -s tests -v

# Run only frozen-training checks.
.venv/bin/python -E -m unittest discover -s tests -p 'test_frozen_training.py' -v
```

Fresh setup only (do not clone over the existing `BiRefNet/` folder):

```bash
git clone https://github.com/ZhengPeng7/BiRefNet.git BiRefNet
git -C BiRefNet checkout ebcc0bc8ec7fe919cec829f2dea656b3078acddc
python3.12 -m venv .venv
.venv/bin/python -E -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -E -m pip install -r requirements.txt -c requirements.lock.txt
.venv/bin/python -E -m pip install -r requirements-training.txt -c requirements.lock.txt
.venv/bin/python -E download_model.py

# Camera environment requires Python 3.10 or 3.11; this setup uses 3.10.
python3.10 -m venv .venv-camera
.venv-camera/bin/python -E -m pip install -r requirements-camera.txt
```

## Complete project-script option reference

Paths below are relative to the project root. All scripts in this table accept
`-h` / `--help`. Flags such as `--train` and `--auto` are off unless supplied.

| Script | Options and defaults |
|---|---|
| `capture_l515.py` | `--output input/l515`; `--serial` unset; `--list`; `--width 1920`; `--height 1080`; `--fps 30`; `--interval 1.0`; `--auto`; `--count 0` (unlimited); `--warmup 2.0`; `--no-preview` (requires `--auto`) |
| `update_dataset.py` | `--input input/l515`; `--base output/l515_emergency_stop`; `--dataset output/l515_dataset`; repeatable `--val-session` (none); `--dry-run`; `--train`; `--model runs/segment/emergency_stop_backgrounds/weights/best.pt`; `--epochs 100`; `--batch 4`; `--device 0`; `--freeze 11`; `--optimizer AdamW` (`AdamW`/`SGD`); `--lr0 0.0001` |
| `train_backgrounds.py` | `--data` latest dataset or original fallback; `--backgrounds backgrounds/picsum`; `--background-probability 0.85`; `--model yolo11n-seg.pt`; `--epochs 100`; `--batch 4`; `--imgsz 640`; `--workers 2`; `--device 0`; `--seed 42`; `--name emergency_stop_backgrounds`; `--freeze 0`; `--optimizer auto` (`auto`/`AdamW`/`SGD`); `--lr0` unset; `--preview 0`; `--preview-output output/background_preview` |
| `test_l515.py` | `--weights runs/segment/emergency_stop_backgrounds/weights/best.pt`; `--camera-python .venv-camera/bin/python`; `--serial` unset; `--list`; `--width 1280`; `--height 720`; `--fps 30`; `--imgsz 640`; `--conf 0.5`; `--iou 0.7`; `--device auto`; `--no-preview`; `--max-frames 0` (unlimited); `--save-every 0` (manual); `--output output/l515_test` |
| `remove_background.py` | Required positional `input`; `--output output`; `--model models/birefnet`; `--device auto` (`auto`/`cuda`/`cpu`); `--size 1024` (at least 256, multiple of 32); `--threshold 0.5`; `--min-area 100`; `--epsilon 0.001`; `--class-name` unset |
| `prepare_emergency_stop.py` | `--raw output/l515_100`; `--output output/l515_emergency_stop` |
| `download_backgrounds.py` | `--output backgrounds/picsum`; `--count 100`; `--seed 42`; `--width 1280`; `--height 720` |
| `l515_camera_worker.py` (internal) | `--width 1280`; `--height 720`; `--fps 30`; `--serial` unset; `--list` |

`download_model.py` has no argument parser: even `--help` would run the download.
`background_augmentation.py` and `l515_protocol.py` are imported support modules,
not standalone commands. Upstream `BiRefNet/` research scripts are outside this
project-wrapper reference.

```bash
# Show supported options without running the corresponding workflow.
.venv-camera/bin/python -E capture_l515.py --help
.venv/bin/python -E update_dataset.py --help
.venv/bin/python -E train_backgrounds.py --help
.venv/bin/python -E test_l515.py --help
.venv/bin/python -E remove_background.py --help
.venv/bin/python -E prepare_emergency_stop.py --help
.venv/bin/python -E download_backgrounds.py --help
.venv-camera/bin/python -E l515_camera_worker.py --help

# Camera-worker diagnostic only. Normal streaming is managed by test_l515.py.
.venv-camera/bin/python -E l515_camera_worker.py --list
```

## Project session handoff

In the Codex conversation, send `/handoff` or invoke `$handoff` to save progress
in `HANDOFF.md`. These are chat instructions, not Bash commands. A new session
opened in this project is instructed by `AGENTS.md` to recap the saved state.

For pipeline details and known model limitations, see [README.md](README.md).
