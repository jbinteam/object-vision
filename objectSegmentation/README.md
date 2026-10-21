# BiRefNet background removal

Taking over this project? Start with the [junior handover guide](docs/JUNIOR_HANDOVER.md)
and [two-project handover](../objectScanner/PROJECT_HANDOVER.md) for setup,
dataset/model provenance and the connection to objectScanner.

For copyable commands, workflow examples, and all project CLI options, see the
[CLI cheat sheet](CLI_CHEATSHEET.md).

This workspace contains the official [BiRefNet repository](https://github.com/ZhengPeng7/BiRefNet)
in `BiRefNet/`, an isolated `.venv`, and a batch wrapper for dataset preparation.
The upstream checkout is unmodified.

## Capture images with the RealSense L515

The camera script uses a separate `.venv-camera` environment because L515 support
requires an older SDK. The [RealSense compatibility table](https://github.com/realsenseai/librealsense/releases)
lists SDK 2.50.0 as validated and 2.54.2 as supporting L515 without validation.
This setup pins `pyrealsense2==2.54.2.5684` with Python 3.10; its wheels do not support
the model environment's Python 3.12. The camera environment is already installed locally. The `-E` flag ignores inherited
Python paths (including the shell's ROS Python 3.12 paths).

Connect the L515 with a USB 3 data cable, close other camera applications, and run:

```bash
.venv-camera/bin/python -E capture_l515.py --list
.venv-camera/bin/python -E capture_l515.py
```

The preview opens at 1920 × 1080 / 30 FPS capture resolution. Controls:

- **Space** or **S**: save one color image.
- **R**: toggle automatic capture (one image per second by default).
- **Q**, **Esc**, or closing the window: stop. Ctrl+C also stops capture.

There is a two-second exposure warmup before saving. Each run creates a unique
`input/l515/<session>/` folder with full-resolution color PNGs, camera details in
`session.json`, and per-image timestamps in `frames.jsonl`. Preview text is never
drawn on saved images. Depth is not recorded; BiRefNet takes color images.

Capture 100 images at two-second intervals (add `--no-preview` for headless use):

```bash
.venv-camera/bin/python -E capture_l515.py --auto --interval 2 --count 100
```

Use `--serial SERIAL` if multiple L515 cameras are connected. `--list` prints the
available color modes; select one with `--width`, `--height`, and `--fps`, for example
`--width 1280 --height 720 --fps 30`. Unsupported modes produce an error.

Process a captured session using the model environment:

```bash
.venv/bin/python remove_background.py input/l515/SESSION_FOLDER \
  --output output/my_object --class-name bottle
```

To recreate the camera environment, use a Python 3.10 or 3.11 interpreter:

```bash
python3.10 -m venv .venv-camera
.venv-camera/bin/python -E -m pip install -r requirements-camera.txt
```

If `--list` cannot see the connected camera, check USB connectivity and the
[SDK 2.54.2 Linux udev setup](https://github.com/realsenseai/librealsense/blob/v2.54.2/doc/installation.md).
No system drivers, firmware, or udev rules were changed by this setup. The L515 was
not connected during development: SDK import/device discovery and simulated capture
are checked locally; live streaming and preview still require the physical camera.

## Run

From this directory:

```bash
source .venv/bin/activate
python remove_background.py /path/to/photos --output output/my_batch
```

An input can be one image or a directory, searched recursively. The output directory
must be empty or new. CUDA is selected automatically; `--device cpu` is also available.
The default is 1024 × 1024 inference with CUDA FP16 autocast. Outputs are restored to
the input dimensions after EXIF orientation correction.

Each image produces:

- `cutouts/`: transparent RGBA PNGs for background removal/compositing.
- `alpha/`: soft foreground probabilities as 8-bit grayscale PNGs.
- `masks/`: binary foreground masks (default threshold 0.5).
- `previews/`: foreground tint with polygon outlines for review.
- `manifest.json`: sources, parameters, dimensions, region counts, and review warnings.

Names retain the source extension: `item.jpg` becomes `cutouts/item.jpg.png`.
Relative subdirectories are preserved, so duplicate names do not overwrite each other.

## Draft YOLO segmentation labels

For photos whose foreground objects all belong to one known class:

```bash
python remove_background.py /path/to/photos \
  --output output/my_class --class-name bottle
```

This also saves oriented RGB images in `images/` and matching `labels/*.txt` files.
Every label uses class ID **0** with the supplied class name recorded in the manifest.
Coordinates follow the [Ultralytics segmentation format](https://docs.ultralytics.com/datasets/segment/).
Do not combine multiple runs with different class names without remapping class IDs.

BiRefNet predicts foreground, not semantic classes or separate object instances.
Each external connected region becomes one candidate polygon; touching objects can
merge, and disconnected parts of one object can split. Holes remain in the masks but
are omitted from the polygon labels. Empty masks produce empty label files and a
review warning. These are draft annotations: inspect `previews/` and the manifest
before training. Single foreground objects per image are the simplest starting point.

`--threshold 0.5`, `--min-area 100` (original-image pixels), and `--epsilon 0.001`
(fraction of contour perimeter) control mask thresholding and polygon conversion.
Area filtering affects labels, not the saved masks or cutouts.

Dataset splitting, class remapping, background compositing, and YOLO training are
subsequent steps. No train/validation split or training configuration is fabricated
without your images and class definitions. Split by original object/capture session
before augmentation to keep related images out of both training and validation.

## Prepared L515 emergency-stop dataset

The captured 100-image batch is processed in `output/l515_emergency_stop/`.
Class **0 = emergency_stop**, covering the red button and yellow/black enclosure,
excluding the cable. Original captures and the first-pass BiRefNet results in
`output/l515_100/` are preserved.

`prepare_emergency_stop.py` performs a dataset-specific refinement: it finds the
yellow enclosure surrounding the red button, runs BiRefNet on a tighter crop,
removes red wire pixels while protecting the button, and removes thin connections
and unrelated components. It is intended for this object/color combination.

```bash
.venv/bin/python -E prepare_emergency_stop.py --output output/emergency_stop_new
```

Outputs include original `images/`, YOLO polygon `labels/`, binary `masks/`,
transparent `cutouts/`, and full-resolution `previews/`. `review/sheet_*.jpg` shows
object crops for boundary checks; `review/full_*.jpg` shows complete frames.
These are draft automatic annotations, not manually traced ground truth.

`data.yaml` defines the dataset with capture-session separation: the first session
is training and the second is validation. There are 49 training and 50 validation
images; first-session frame 41 is held out because a second emergency-stop assembly
is partly visible at the top edge and only the main object is labeled. All 100
images and primary-object labels are retained, with that frame flagged in the manifest.
Both sessions show the same object in a similar setting, so this validation set
does not establish generalization to new objects or environments. The completed
100-epoch training run is in `runs/segment/emergency_stop_backgrounds/`.

## On-the-fly background augmentation and training

100 random landscape photos are downloaded from [Lorem Picsum](https://picsum.photos/)
to `backgrounds/picsum/`. `sources.json` records photographer names, original source
links, download URLs, dimensions, and checksums. The downloader uses a fixed sampling
seed and checks image decoding; rerunning it reuses completed downloads.

```bash
.venv/bin/python -E -m pip install -r requirements-training.txt -c requirements.lock.txt
.venv/bin/python -E download_backgrounds.py --count 100
```

`background_augmentation.py` extends the pinned Ultralytics 8.3.253 segmentation
dataset/trainer. Every training read has an **85% chance** of receiving a freshly
sampled background, with random background cropping and flipping. The remaining
15% retain the original scene. Foreground pixels come from the original RGB images
and the refined binary masks; background replacement does not move the object or
change polygon coordinates. YOLO then applies rotation (±15°), scale, translation,
horizontal flipping, and color augmentation to the image and labels together.
The default disables mosaic/mixup/copy-paste so the small object remains visible.

This happens inside the loader, including on later epochs and with multiple workers;
there is no static expanded training dataset. Validation uses the **50 original
photos**, with no random background replacement. The held-out review image remains
excluded. `--background-probability 1` replaces every training background; `0`
retains every original background.

Inspect the already-generated examples in `output/background_preview/`, or make
another preview through the actual training transforms:

```bash
.venv/bin/python -E train_backgrounds.py --preview 12 \
  --preview-output output/background_preview_new
```

The first four examples reuse the same source image to demonstrate fresh variation.
`sample_*.jpg` are composites; `sample_*_labels.jpg` and `contact_*.jpg` overlay the
training masks. Preview files are not included in training.

Start a full GPU training run:

```bash
.venv/bin/python -E train_backgrounds.py \
  --model yolo11n-seg.pt --epochs 100 --batch 4 --imgsz 640 \
  --workers 2 --device 0 --name emergency_stop_backgrounds
```

Results and weights go into `runs/segment/<run-name>/`; each run also records
`background_augmentation.json`. Use this Python entry point for background
replacement; a plain `yolo train` command does not install the custom loader.
Only a single GPU or CPU is supported by this wrapper. The default batch size is
conservative for the local 8 GB GPU. Random photos improve appearance variety but
do not add new physical views, object shapes, or realistic lighting interactions.

Verification: compositor/dataset tests check unchanged foreground pixels, fresh
background sampling, unchanged polygons, validation isolation, and missing-mask
errors. `runs/segment/background_smoke/` contains a completed one-epoch GPU test
with two loader workers, 49 training images, and 50 validation images. Its weights
are for pipeline verification, not a finished trained model.

## Add more captures and retrain (repeatable workflow)

Run these commands from the project directory. Each capture invocation creates a
new session under `input/l515/`; keep these session folders together under that root.

```bash
# Capture another batch (skip this if the photos are already captured):
.venv-camera/bin/python -E capture_l515.py --auto --interval 2 --count 30

# Discover new images, generate cable-excluded masks/YOLO labels, and publish a dataset:
.venv/bin/python -E update_dataset.py

# Update, then fine-tune best.pt with the YOLO11 backbone frozen:
.venv/bin/python -E update_dataset.py --train
```

The update scans `input/l515/<session>/*.png` and imports the original 100 labeled
images from `output/l515_emergency_stop/` on its first run. BiRefNet runs only for
new or previously failed images. The latest complete dataset is available at
`output/l515_dataset/latest/data.yaml`; the link points to a separate timestamped
version under `output/l515_dataset/versions/`. Older datasets and model checkpoints
are preserved, and the original captured images are never edited.

The current addition is 30 images from session
`20260923T055454_605148Z_a4bb12dd`. With these added, the usable split is **79 training
images and 50 validation images**, with the original ambiguous frame still excluded.
The new session goes entirely into training. This preserves session separation, but
the existing validation set does not independently measure the new perspective.

Review the updated sessions in `output/l515_dataset/latest/review/crops_*.jpg` and
`full_*.jpg`, or inspect full-resolution `previews/`. `manifest.json` records source
and decoded-image checksums, fixed session assignments, provenance, and review flags.
`validation.json` reports checks and excluded frames. The masks are automatic draft
annotations of the red button and yellow/black enclosure, excluding cables; visual
inspection is still useful around the dark base and cable attachment.

Updates check image integrity, mask dimensions, binary mask values, cutout alpha,
normalized polygon coordinates, polygon/mask agreement, and split separation.
Exact decoded-image duplicates are excluded, even if renamed or recompressed as
PNG. An uncertain or failed detection is excluded and reported, never silently
converted into a negative example. Failures are retried on the next update.
New viewpoints still need a visible red button and yellow enclosure for this
object-specific labeling method. Do not put negative-only scenes through it.

Rerunning without new captures reuses the current version without labeling again.
Completed intermediate annotations are cached, so interrupted updates reuse their
work; a new version becomes `latest` only after its checks pass. Previously imported
files must keep their contents; put revised images in a new session. Published
dataset versions should be treated as read-only.

```bash
# List pending work without running BiRefNet:
.venv/bin/python -E update_dataset.py --dry-run

# Reserve an entirely new session for validation (repeat the flag for more sessions):
.venv/bin/python -E update_dataset.py --val-session YOUR_NEW_SESSION_FOLDER_NAME

# Fine-tune from a particular later checkpoint:
.venv/bin/python -E update_dataset.py --train \
  --model runs/segment/YOUR_PREVIOUS_RUN/weights/best.pt --epochs 100 --batch 4
```

Choose `--val-session` on a session's first import. Session assignments persist;
existing training sessions cannot be moved to validation by this command. Otherwise,
new sessions default to training. Near-duplicate video frames are not detected by
pixel checksums, which is another reason to retain session separation.

`--train` starts a new run with fresh optimizer state, initialized by default from
`runs/segment/emergency_stop_backgrounds/weights/best.pt`. It trains on all current
training images with the existing 85% on-the-fly background replacement. It uses
100 epochs, batch 4, GPU 0, **`--freeze 11 --optimizer AdamW --lr0 0.0001`** by default;
`--device cpu` is also supported. In this YOLO11 model, modules 0–10 are the complete
backbone (including SPPF and C2PSA). The remaining neck/head modules learn from the
expanded dataset. Frozen BatchNorm statistics also remain fixed in the installed
Ultralytics version. `freeze_audit.json` records parameter counts and verifies that
the frozen weights and buffers are unchanged at the end of training.

Use `--freeze 0` to train all normally trainable parameters or a smaller positive
number to freeze only the earlier modules. Freezing preserves existing features,
but does not guarantee better accuracy; a substantially different domain may need
more of the backbone to adapt. Keep the old training images in the expanded dataset.

The learning rate is explicit: the wrapper uses AdamW because Ultralytics's `auto`
optimizer overrides a supplied initial learning rate. Bias warmup starts at zero
instead of the upstream 0.1, then ramps toward the requested small learning rate.
The model's fixed DFL projection remains frozen even with `--freeze 0`.

For the equivalent direct training command:

```bash
.venv/bin/python -E train_backgrounds.py \
  --model runs/segment/emergency_stop_backgrounds/weights/best.pt \
  --freeze 11 --optimizer AdamW --lr0 0.0001 \
  --epochs 100 --batch 4 --name emergency_stop_frozen_backbone
```

Pass the newly trained run's `best.pt` with `--model` on a later cycle when you want
to continue from that checkpoint. The direct `train_backgrounds.py` command retains
its general-training defaults (base pretrained model, no user-frozen layers,
automatic optimizer); `update_dataset.py --train` supplies the fine-tuning defaults.
The freeze count refers to top-level model modules, so check the architecture if
using a different model family. See [Ultralytics training arguments](https://docs.ultralytics.com/modes/train/).

This workflow does
not resume the old optimizer or overwrite its weights. The resolved dataset version
is pinned for the run, even if a later update changes `latest`. BiRefNet exits before
YOLO starts, releasing its GPU memory. Running `--train` again intentionally starts
another training run, even when there are no new captures.

Runs are named `runs/segment/emergency_stop_<dataset-version>/` (Ultralytics adds a
suffix if needed). To test that run with the L515:

```bash
.venv/bin/python -E test_l515.py --weights runs/segment/YOUR_NEW_RUN/weights/best.pt
```

`train_backgrounds.py` also defaults to the latest incremental dataset when one
exists; use `--data` to select a specific older dataset. Automated tests cover
incremental updates, repeat runs, session assignments, duplicate exclusion, changed
sources, failed-detection exclusion, and recovery after an interrupted publication.
All 31 project tests passed, including frozen-weight and BatchNorm-state checks.
A one-epoch GPU training check on the expanded dataset
completed in `runs/segment/incremental_dataset_smoke/`, using 79 training images,
50 validation images, and 85% background replacement. This is a pipeline check;
start another retraining run with `update_dataset.py --train`.

Completed frozen-backbone fine-tuning is in
`runs/segment/emergency_stop_frozen_backbone/weights/best.pt`. It loaded the original
trained checkpoint, used the 79/50 expanded dataset and the defaults above, and
stopped at epoch 53 of a requested 100 (patience 25; best epoch 28). The freeze audit
confirmed all 1,365,472 frozen backbone parameters and their BatchNorm state remained
unchanged; 1,477,315 parameters were trainable. These validation images remain from
the original validation session, so this result alone does not establish improved
performance in new environments. To use this checkpoint for the next update:

```bash
.venv/bin/python -E update_dataset.py --train \
  --model runs/segment/emergency_stop_frozen_backbone/weights/best.pt
```

## Test the trained model live with L515

Connect the L515 and close other programs using its camera, then run:

```bash
.venv/bin/python -E test_l515.py
```

This loads `runs/segment/emergency_stop_backgrounds/weights/best.pt`, uses the GPU
when available, and displays live segmentation masks, boxes, confidence scores,
and processing FPS. It uses RGB at 1280×720, 30 FPS, with inference size 640 and
confidence threshold 0.50. Processing FPS excludes display and snapshot writing.
The camera's Python 3.10 worker starts automatically from `.venv-camera`; inference
runs in `.venv`. Both environments from the setup above are required.

Controls: **S / Space** saves a snapshot, **M** toggles the mask overlay,
**+ / −** changes confidence by 0.05, and **Q / Esc** exits. Ctrl+C also stops it.

```bash
# Select a different checkpoint or confidence threshold:
.venv/bin/python -E test_l515.py --weights runs/segment/emergency_stop_backgrounds/weights/best.pt --conf 0.7

# List connected devices and supported RGB modes:
.venv/bin/python -E test_l515.py --list

# Run without a display and save every 15th prediction:
.venv/bin/python -E test_l515.py --no-preview --max-frames 30 --save-every 15
```

Use `--device cpu` for CPU inference, `--serial SERIAL` to select a camera, or
`--width 1920 --height 1080 --fps 30` for full-resolution RGB. `--help` lists options.
Inference uses the real camera image without background replacement.

Each session in `output/l515_test/` records the checkpoint checksum, camera
settings, per-frame detection counts/confidences/timings, and a completion summary.
Each snapshot saves original `color.png`, annotated `prediction.png`, one binary
full-resolution `mask_*.png` per detected instance, and `predictions.json` with
normalized boxes/polygons. These are model predictions, not ground-truth labels.

Verified on the connected L515 with 30 live frames and two saved snapshots. Review
`output/l515_test/20260923T054318_889054Z_913d01e5/snapshot_000030/prediction.png`:
the model incorrectly labeled a filament spool as `emergency_stop` at about 0.51
confidence. This verifies the camera/inference pipeline, not model accuracy;
additional real scenes and negative examples are needed to assess generalization.

## Reproduce the environment

Python 3.12 is used locally. The downloaded code revision is
`ebcc0bc8ec7fe919cec829f2dea656b3078acddc`. The general-purpose model is
`ZhengPeng7/BiRefNet`, pinned at `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`.

```bash
git clone https://github.com/ZhengPeng7/BiRefNet.git BiRefNet
git -C BiRefNet checkout ebcc0bc8ec7fe919cec829f2dea656b3078acddc
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_model.py
```

`requirements.lock.txt` records the validated installed versions. On this Linux/CUDA
platform, install the same PyTorch pair/index first, then use the lock file to recreate
the rest of the environment. The upstream requirements include research/training tools;
this setup has been validated for inference only. Local weights live in `models/birefnet/`.
Inference loads the cloned implementation and local safetensors, without downloading
or executing model code from Hugging Face.

```bash
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
```

## Verified locally

Setup verification passed on the NVIDIA GeForce RTX 4070 Laptop GPU (8 GB),
using PyTorch 2.8.0+cu128. All four polygon-conversion tests passed, and `pip check`
reported no broken requirements. A real 1024-pixel GPU inference run produced a
full-resolution mask, transparent cutout, preview, and valid normalized YOLO label.
The batch loop took 2.71 seconds for the sample, including first inference and PNG
writing, excluding model loading; this is a smoke test, not a throughput benchmark.

The sample is the [PyTorch Hub dog image](https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg).
Input: `input/smoke/dog.jpg`. Results: `output/smoke/`.
