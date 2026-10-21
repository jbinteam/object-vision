# Turntable RGB-to-pose dataset

The collector uses the Orbbec Gemini 305 color image as a model input and saves the object's
**box-center pose** in the rectified color optical frame as its label. It does not
use ICP or YOLO to generate labels. The old `my_old_work/ws_setting.py` projected
fiducial-based workspace corners but stored them in camera coordinates; the new
collector fixes the object frame to a permanent turntable tag frame so labels
remain valid as the camera and turntable move.

New labels name that optical frame `camera_color_optical_frame`, matching ROS.
Older labels may say `color_optical_frame`; they contain coordinates in the same
rectified color optical convention. The raw scan PLY is in the first scanning
camera's optical frame, while the cropped reference PLY is in `box_center`.
To compare the raw scan with a later observed cloud, apply
`T_camera_object @ inverse(T_scan_object)` to raw scan points. To compare the
cropped reference with an observed cloud, apply `T_camera_object`.

## Physical layout

Use one AprilTag family for the whole session. The collector supports
`tag36h11` and the `tagStandard41h12` family used by
`my_old_work/generate_apriltags.py` and
`my_old_work/detect_tagStandard41h12.py`. Never mix families in one session.
This setup uses two measured tag edge sizes: the four temporary box tags are
14 mm and the permanent table tags are 28 mm. Measure the detected tag square
itself, excluding the added white quiet-zone margin and the rest of the page.

- Temporary setup IDs `1,2,3,4` have no fixed order. Put **any three** tag
  centers on three corners of the rectangular base on the turntable surface.
  Put the remaining tag on the object's top surface. The collector detects the
  elevated tag, finds the right-angle base corner, and sorts the two adjacent
  base corners automatically. Small deviations from 90 degrees are
  orthogonalized. The top tag's horizontal position does not set box center.
- Permanent tags `5,6,7,8,9,10` remain rigidly fixed on the same table plane.
  Add more unique IDs with `--permanent-ids` if useful. Only **two** need to be
  visible when setting the box; the others can be mapped later. Arrange them
  around the object so nearby views share at least two visible mapped tags.
  At least two mapped tags must be visible in every captured image.

All four temporary tags are needed only for **Set object BB**. Once clicked,
remove all four, including the one on top of the object. The saved box is then
carried by the permanent turntable tags in later camera views.

To adjust the box later, put all four temporary tags back around the same
table-fixed object and keep at least two mapped permanent tags visible. Click
**Reconfig BB [R]** or focus the window and press `R`. The collector keeps the
permanent table map, resolves the new box, increments the calibration revision,
and updates every existing JSON pose label to use the new box frame. Before
changing anything, it archives the prior calibration and labels under
`reconfiguration_history/`. Remove all four temporary tags again before capture.
This relabeling assumes the object itself has not moved relative to the table
since the earlier images. If the object was physically repositioned, start a
new output directory instead of reconfiguring the old session.

The object must not shift relative to the table after the box is set. The pose
label uses the automatically resolved box axes; for an asymmetric object, put
the three surface tags on three corners of the exact base box you want the
model to learn. A top tag measures height. The permanent tag layout anchors the
resolved box after all four temporary tags are removed.

## Test tag detection first

Test the detector on one generated or photographed image without opening the
camera:

```bash
.venv/bin/python scripts/test_apriltags.py \
  --image apriltags/tagStandard41h12/tagStandard41h12_00010.png \
  --family tagStandard41h12 \
  --output output/tag_test.png \
  --no-display
```

The command prints the detected IDs and saves an annotated image. In the
annotation, corner indices `0,1,2,3` show the order used for pose estimation.
Next, test the printed tags through the Gemini 305:

```bash
.venv/bin/python scripts/test_apriltags.py \
  --camera --camera-backend orbbec \
  --family tagStandard41h12 \
  --setup-tag-size-mm 14 \
  --permanent-tag-size-mm 28
```

The live view displays tag IDs, corner indices, axes, distance, and reprojection
error. Press `S` to save both the raw and annotated frames under
`output/apriltag_test/`; press `Q` or Escape to quit. Rotate and tilt each print,
move it toward the image edges, and test the farthest intended camera distance.
The detected ID must stay correct. A small reprojection error is preferred;
large or unstable error usually indicates blur, glare, an incorrect printed
size, or a tag that occupies too few pixels.

For `tag36h11` prints, replace the family argument with `tag36h11`.

## Collect the dataset

From the project root, with the Gemini 305 connected and no other process using it:

```bash
scripts/collect_process/01_capture.sh
```

This starts the `my_object_gemini305` session with category `my_object`. Run
subsequent collection scripts with the same session/category; see the
[step-by-step collection guide](../scripts/collect_process/README.md).
The equivalent direct collector command is:

```bash
.venv/bin/python scripts/capture_turntable_dataset.py \
  --name my_object_gemini305 \
  --camera-backend orbbec \
  --centerpose-category my_object \
  --centerpose-split train \
  --tag-family tagStandard41h12 \
  --setup-tag-size-mm 14 \
  --permanent-tag-size-mm 28 \
  --setup-ids 1,2,3,4 \
  --permanent-ids 5,6,7,8,9,10
```

`--name NAME` stores the session under `datasets/NAME`. Use `--output PATH`
instead when you need an explicit directory elsewhere; the two options are
mutually exclusive.

Start a fresh session when switching from L515 to Gemini; the collector rejects
a different camera serial/backend for an existing calibration. Add `--serial
SERIAL` when multiple cameras are connected. Orbbec collection defaults to a
0.04–1.5 m software depth interval, including diagnostic scans; override it with
`--depth-min` and `--depth-max`. The legacy `--camera-backend realsense` option
retains a 0.25 m near cutoff (`auto` also uses 0.25 m). These are depth filters,
not a guarantee of physical accuracy at their limits. See the
[camera guide](GEMINI305_CAMERA.md) for calibration and validation details.

Every capture also writes a CenterPose/Objectron-style same-basename pair to
`datasets/NAME/centerpose/outf/CATEGORY_SPLIT/`. Its JSON contains `camera_data`
and one `objects` entry with the center-first nine-point `projected_cuboid`,
OpenGL-camera pose, 3D keypoints, and scale. Export converts this collector's
Z-up object frame to CenterPose's Y-up convention. The directory under `outf/`
can be copied directly into CenterPose's `data/outf/` directory.

The collector establishes the canonical frame from the image workflow first.
The production reference is then fused from the captured, masked RGB-D views
with their saved poses:

1. Place the setup and permanent tags, then press **Set BB [S]**. This saves the
   single canonical `box_center` origin, axes and dimensions.
2. Remove the setup tags and capture all desired image views with **Capture
   [SPACE]**. The permanent table tags keep every image pose in `box_center`.
3. Generate and review foreground masks with `02_generate_masks.sh`, then export
   the paired YOLO labels with `03_export_yolo.sh`.
4. Run `04_build_reference.sh`. It integrates each masked depth view with its
   saved `T_camera_object` pose, crops the fused TSDF to the calibrated metric
   box plus 3 mm, and writes `references/CATEGORY/reference.ply` directly in
   `box_center`. The PnP keypoints use the same calibration and no reference
   axis is manually assigned.

**Scan PLY [P]** remains available for diagnostic moving-camera scans. Hold at
least two mapped permanent tags visible until anchoring succeeds, then move the
camera slowly while keeping the object and table fixed. Its raw PLY and Web UI
crop are not used by the production PnP→GICP path. The diagnostic editor can be
reopened with `01b_edit_reference.sh SESSION CATEGORY` after the collector exits.

Collector reference scans default to CUDA, a 3 mm TSDF grid, full-resolution
RGB-D, and a 60-second per-operation watchdog. This matches the successful
anchored GPU scan for the current object. CUDA must be available in the
collector's Python environment. To use CPU on another machine, pass
`--scan-device cpu --scan-downscale 2`; CPU raycasting can take longer than the
scanner's generic 15-second watchdog on its first fused frame. Override
`--scan-voxel-mm` and `--scan-stall-timeout` if needed.

Diagnostic TSDF scans are retained under `datasets/NAME/reference_scans/`;
the calibrated RGB-D TSDF reference bundle is under
`datasets/NAME/references/CATEGORY/`. Press **Inspect scan [E]** to open the
diagnostic editor. Use `--reference PATH` to resume with an already prepared
canonical PLY. The production reference and all cuboid labels use
the same convention, `P_camera = T_camera_object * P_object`.

Existing samples can be synchronized without starting the camera:

```bash
.venv/bin/python scripts/export_centerpose_dataset.py datasets/my_object \
  --category my_object --split train
```

Keep one capture session in one split. For a meaningful validation set, record
a separate session with `--centerpose-split val` rather than splitting adjacent
frames from the same turntable sweep.

## BiRefNet foreground masks for on-the-fly augmentation

The CenterPose training pairs can be expanded with the cable-free emergency-stop
refinement and background library from the sibling `objectSegmentation` project.
Run this with that project's model environment because it contains BiRefNet and
CUDA PyTorch:

```bash
../objectSegmentation/.venv/bin/python -E \
  scripts/augment_centerpose_backgrounds.py \
  --dataset datasets/my_object \
  --output datasets/my_object/centerpose_augmented
```

For each source image, the script runs BiRefNet on a pose-guided crop, uses the
calibrated projected cuboid to exclude unrelated foreground, subtracts recorded
tag-inpaint regions, breaks connections through bright neutral paper/background
and thin cables, selects the component at the projected object center, and
validates the result against the projected 3D box. Full-frame BiRefNet is retained
as a fallback. The default produces reviewed masks and previews only—no synthetic
RGB files. YOLO samples a new background at training time, while preserving the
captured foreground pixels and image geometry, so both segmentation and ordered
keypoint labels remain exact. `--augmentations-per-image N` remains available
only for legacy CenterPose experiments that require materialized PNG/JSON pairs.
RGBA cutouts are also omitted by default; request them with `--write-cutouts`
only when another tool needs them.

The output is built in a temporary directory and published only after every mask
succeeds. Inspect `centerpose_augmented/review/` before using
the data; green is the selected mask, red is its contour, and blue is the
projected pose box. `manifest.json` records mask checks and the exact background
used for every generated image when legacy materialization is requested.
Augmentation is intentionally training-only;
validation should use a separate, unmodified capture session.

## Train the NVlabs CenterPose benchmark model

The training launcher uses the official `dlav1_34` DLA-34 + convGRU CenterPose
architecture while bypassing the upstream entry point's hard-coded `bike`
category. It creates a lightweight symlink dataset, so the 515 MB augmentation
is not duplicated. Validation is grouped by original camera view: every
synthetic variant of a held-out view is excluded from training, and validation
uses only that view's original capture.

Prepare and inspect the deterministic split without installing CenterPose:

```bash
.venv/bin/python scripts/train_centerpose.py --prepare-only
```

With 64 source views and the default 12.5% holdout, this produces 616 training
pairs and 8 validation pairs under
`datasets/my_object_contrast/centerpose_training/outf/`.

This workstation now has a dedicated `.venv-centerpose` environment and the
official repository at `../CenterPose`. The upstream repository is licensed for
non-commercial use; review its `LICENSE.md` before training. To recreate the
tested setup on this machine, use Python 3.12, install the pinned CUDA/PyTorch
dependencies, and apply the compatibility patch:

```bash
python3.12 -m venv .venv-centerpose
.venv-centerpose/bin/python -m pip install --upgrade pip setuptools wheel
.venv-centerpose/bin/python -m pip install -r requirements-centerpose.txt
git clone --recursive https://github.com/NVlabs/CenterPose.git ../CenterPose
git -C ../CenterPose apply \
  ../objectScanner/patches/centerpose-modern-pytorch.patch
```

The compatibility patch replaces CenterPose's PyTorch-1.1-era DCNv2 extension
with `torchvision.ops.deform_conv2d` and updates its removed scikit-learn linear
assignment import. Do not run the old DCNv2 `make.sh` with this setup.

Run the launcher with `.venv-centerpose`. The defaults use the benchmark
140-epoch schedule with learning-rate drops at epochs 90 and 120. Batch size 8
is a conservative starting point for an 8 GB GPU; the launcher linearly scales
the benchmark learning rate of `6e-5` at batch size 16.

```bash
.venv-centerpose/bin/python scripts/train_centerpose.py \
  --centerpose-root ../CenterPose \
  --dataset datasets/my_object_contrast/centerpose_augmented \
  --batch-size 8 \
  --gpus 0 \
  --save-all
```

Checkpoints, logs, the resolved launch configuration, and TensorBoard data are
written under `training/centerpose/`. Use `--dry-run` to validate the checkout
and print the resolved configuration without training. Set
`NO_ALBUMENTATIONS_UPDATE=1` before the command to suppress Albumentations'
network update check on an offline machine. To resume, pass both `--resume` and
the previous `my_object_last.pth` with `--load-model`.

## Export paired segmentation and 8-keypoint YOLO labels

After reviewing the masks under `centerpose_augmented_clean/review/`, export two
synchronized Ultralytics datasets. Only the captured source images are exported;
the matching binary masks are placed beside both tasks for training-time
compositing:

```bash
.venv/bin/python scripts/generate_yolo_seg_pose_dataset.py \
  --dataset datasets/my_object \
  --augmentation-root datasets/my_object/centerpose_augmented_clean \
  --output datasets/my_object/yolo_seg_pose
```

The exporter writes the same image stems to `segmentation/` and `pose/`. The
segmentation labels contain the reviewed mask contour. The pose labels contain
the mask-derived bounding box followed by eight `(x, y, visibility)` cuboid
corners. All coordinates are normalized in the YOLO text files. Numbered pixel
overlays under `review/` and combined JSON records under `annotations/` make the
pairing auditable. It intentionally ignores old `__bg_*` images unless
`--include-materialized-augmentations` is explicitly requested.

Train with the sibling project's Ultralytics environment:

```bash
../objectSegmentation/.venv/bin/python -E scripts/train_yolo_seg_pose.py \
  --dataset datasets/my_object/yolo_seg_pose \
  --backgrounds ../objectSegmentation/backgrounds/picsum \
  --background-probability 0.85 \
  --task both
```

Each training read independently selects/crops/flips a background. Validation
always uses the original captures. Both segmentation and pose use the same
training-only compositor. Disable it only for a controlled benchmark with
`--no-background-augmentation`.

Polygon conversion deliberately follows the proven `objectSegmentation`
workflow: external contours only, regions below 100 px² ignored, simplification
at 0.001 of contour perimeter, and polygon-to-mask IoU validation at 0.98. For
this single-object pose dataset, more than one surviving external component is a
hard error rather than multiple YOLO instances. Binary masks, source-image
hashes when available, and polygon agreement are checked before any dataset is published. The
thresholds can be changed with `--min-polygon-area`, `--polygon-epsilon`, and
`--min-polygon-mask-iou`.

The immutable keypoint order is:

```text
0 (-X,-Y,-Z)  1 (+X,-Y,-Z)  2 (-X,+Y,-Z)  3 (+X,+Y,-Z)
4 (-X,-Y,+Z)  5 (+X,-Y,+Z)  6 (-X,+Y,+Z)  7 (+X,+Y,+Z)
```

It is also saved with metric object-frame coordinates in
`object_keypoints.yaml`. The convention remains
`P_camera = T_camera_object * P_object`. In-frame projected cuboid corners use
visibility `2`; off-image corners use `(0, 0, 0)` and are never clipped to the
border. Because these are virtual cuboid corners, this is annotation
availability—not literal surface occlusion.

If `centerpose_training/split.json` exists, it is reused automatically: held-out
source views stay out of training and their original captures are used for
validation. Otherwise all samples are assigned to
`train`. Pass `--split-manifest` to select a split explicitly and
`--copy-images` for a portable export; the default uses relative symlinks.

The outputs have separate `segmentation/data.yaml` and `pose/data.yaml` files
because the current runtime uses distinct Ultralytics segmentation and pose
tasks. The pose YAML uses identity `flip_idx`; disable horizontal flip
augmentation (`fliplr=0`) unless mirrored object-frame supervision is knowingly
valid for the object.

## Legacy masked-depth point-cloud fusion

`generate_centerpose_reference.py` is retained for diagnostics and old datasets.
It stitches individually masked RGB-D turntable views without TSDF integration;
the current `build_calibrated_reference.py` uses the same calibrated poses and
reviewed masks but fuses them into a TSDF surface. To inspect legacy
inputs without generating a reference:

```bash
.venv/bin/python scripts/generate_centerpose_reference.py \
  --dataset datasets/my_object_rgbd \
  --augmentation-root datasets/my_object_rgbd/centerpose_augmented_clean \
  --check-only
```

To deliberately reproduce the legacy result:

```bash
.venv/bin/python scripts/generate_centerpose_reference.py \
  --dataset datasets/my_object_rgbd \
  --augmentation-root datasets/my_object_rgbd/centerpose_augmented_clean \
  --output references/my_object_rgbd_centerpose
```

For every valid foreground depth pixel the generator deprojects
`P_camera`, applies `inverse(T_camera_object)`, and obtains `P_object`. Points
inconsistent with the calibrated metric box are rejected. The output contains:

```text
reference.ply                         fused/downsampled reference in box_center
frame.json                            frame contract, source poses and quality metrics
object_keypoints.yaml                 matching eight metric cuboid corners
reference_bundle.zip                  portable reference/configuration bundle
observations/camera_frame/<view>.ply  masked RGB-D observations in camera coordinates
observations/object_frame/<view>.ply  the same observations in box_center
```

Unlike a Web UI export, no manual axis assignment is needed. Every observation is
transformed into the already calibrated object frame before fusion, so the origin
of `reference.ply` is exactly the CenterPose cuboid center and its axes are the
same deterministic axes used by the 3D keypoints, projected 2D corners, PnP,
GICP, and TF. `frame.json` consequently records identity `T_scan_obj` and
`T_obj_scan`; applying either capture pose or the Web UI frame transform to this
reference a second time would be an error.

Prefer `04_build_reference.sh` for the registration reference. The legacy output
remains useful for per-view camera/object-cloud diagnostics.

The existing `datasets/my_object` and `datasets/my_object_contrast` sessions were
captured before aligned depth persistence was added: they currently have zero
depth images and cannot be converted into a metric reference without recapture.

## Next implementation: dense OmniNOCS labels and model

Decision recorded 2026-09-24: finish and evaluate the current CenterPose
benchmark before beginning this milestone. The next implementation will extend
the dataset pipeline toward OmniNOCS-style dense object-coordinate prediction;
it must not alter the CenterPose dataset or interrupt an active training run.

The implementation scope is:

1. Extend capture to save metric L515 depth aligned to the rectified color image,
   with its scale and invalid-pixel convention recorded in each sample.
2. Generate an object instance mask and a 16-bit four-channel NOCS target: XYZ
   object coordinates plus a valid-coordinate channel. For each valid masked
   pixel, deproject depth with the saved intrinsics, transform it with
   `inverse(T_camera_object)`, and normalize it by the object's bounding-box
   diagonal. Confirm the canonical object-axis convention before publishing data.
3. Reject invalid depth, background, tag, occluder, and geometrically inconsistent
   samples. Keep unknown pixels distinct from known background.
4. Reuse each original mask and NOCS target for its background-augmented images,
   because the current compositor leaves foreground geometry unchanged.
5. Investigate rendering targets from an accurately box-frame-aligned mesh as a
   recovery path for the existing captures, which do not contain saved depth.
6. Implement and benchmark a dense-coordinate pose model. Treat NOCSformer as a
   research architecture to reproduce rather than an available drop-in package:
   its official repository publishes the dataset format and annotations but not
   a complete training/inference release. Its inference contract also requires a
   separate 2D object box provider.

Acceptance requires a visual NOCS/mask review set, numerical reprojection checks,
a source-view-grouped train/validation split, and a direct pose-accuracy and
runtime comparison with the completed CenterPose checkpoint on identical held-out
original views.

The CPU-only dataset-generator foundation is prepared. New captures now store
rectified-color-aligned depth as `uint16` millimetres with zero for invalid
pixels, and record its path, scale, alignment, and accepted range in the rich
label. Use the dedicated NOCS-ready collector for the next dataset (do not launch
it while another process is using the camera):

```bash
.venv/bin/python scripts/capture_nocs_dataset.py \
  --name my_object_nocs \
  --camera-backend orbbec \
  --category my_object \
  --split train \
  --tag-family tagStandard41h12 \
  --setup-tag-size-mm 14 \
  --permanent-tag-size-mm 28 \
  --setup-ids 1,2,3,4 \
  --permanent-ids 5,6,7,8,9,10
```

It uses the same calibrated AprilTag workflow and keyboard controls as the
turntable collector. Every Space capture saves RGB, raw RGB, aligned depth, pose,
intrinsics, and metric dimensions. It deliberately does not run BiRefNet or write
dense NOCS maps during acquisition; masks and coordinates are generated and
reviewed afterward.

Check whether a capture session has everything needed without writing an output
dataset:

```bash
.venv/bin/python scripts/generate_nocs_dataset.py \
  --dataset datasets/my_object_contrast \
  --check-only
```

The existing 64-view dataset reports 64 missing depth images because it was
captured before depth persistence was added. Do not fabricate dense targets from
its RGB, mask, and cuboid alone. New depth-enabled captures can be converted with:

```bash
.venv/bin/python scripts/generate_nocs_dataset.py \
  --dataset datasets/NEW_CAPTURE \
  --augmentation-root datasets/NEW_CAPTURE/centerpose_augmented \
  --split-manifest datasets/NEW_CAPTURE/centerpose_training/split.json \
  --output datasets/NEW_CAPTURE/omninocs
```

The generator follows the published OmniNOCS storage layout: a `uint16` instance
PNG and a four-channel `uint16` XYZ-valid PNG per source view, plus per-split JSON
metadata. It deprojects masked depth, applies `inverse(T_camera_object)`, rejects
points inconsistent with the metric box, normalizes by the box diagonal, checks
reprojection numerically, and creates review images. Background variants share
their source view's dense target; held-out validation views keep only the original
RGB image. See the [OmniNOCS paper](https://arxiv.org/abs/2407.08711) and
[official data format](https://github.com/google-deepmind/omninocs#data-format).

For existing `tag36h11` prints, pass `--tag-family tag36h11`. Your generator
can create the selected `tagStandard41h12` IDs with:

```bash
python my_old_work/generate_apriltags.py \
  --families tagStandard41h12 \
  --ids 1,2,3,4,5,6,7,8,9,10 \
  --out apriltags
```

That generator requires its own `moms-apriltag` package. The capture collector
uses `pupil-apriltags`, which is included in the project requirements. Printing
software must preserve equal X/Y scale. Measure the final printed markers and
pass their physical edge lengths to `--setup-tag-size-mm` and
`--permanent-tag-size-mm`; the generator's `--size` argument controls pixels,
not millimetres.

The window displays tag IDs. Once all four setup tags and at least two
permanent tags are detected, click **Set object BB** or press `S`. Check that
the projected box follows the object, then remove all four setup tags. Rotate the
table or move the camera until an unmapped permanent tag is visible alongside
at least two already mapped tags. Stop both and click **Add tags** or press `A`.
Repeat this as needed until the permanent tags you plan to use are mapped.
The saved calibration grows in place; the object frame stays fixed. You can
capture views during this process whenever two mapped tags are visible.

Click **Capture [SPACE]** or focus the camera window and press Space after each
camera/table move. The tool refuses
a capture if fewer than two mapped permanent tags are visible, reprojection
error exceeds 3 pixels, the box is outside the image, or a setup tag remains.
Press `Q` to quit. Re-run the same command to resume a calibrated session.
For other printed IDs, pass four IDs to `--setup-ids` and a comma-separated
`--permanent-ids` containing **all** permanent IDs; use the same IDs and both
tag sizes when resuming. If a new tag never overlaps two mapped tags in one view,
move the camera farther back temporarily to register it, or rearrange the
tags before collecting data. Avoid moving any tag relative to the table after
calibration.

The output contains:

```text
calibration.json             fixed T_table_object, box dimensions, permanent tag layout
images/<timestamp>.png      RGB training image with permanent tags inpainted
raw_images/<timestamp>.png  original, unannotated RGB frame for audit
depths/<timestamp>.png      rectified-color-aligned uint16 depth in millimetres
labels/<timestamp>.json     pose and projection label for that image
samples.jsonl               relative training image/label paths
```

Each label has `T_camera_object` (4×4, metres) and `center_xyz_m` for 3D pose,
`rotation_vector_rad` for orientation, `center_uv_px`, projected 3D box corners,
and clipped `bbox_xyxy_px_clipped` for 2D training. It also records the color
intrinsics, image size, detected permanent IDs, tag reprojection error, and
capture time. `T_camera_object` maps an object-frame point into the rectified
color optical frame. The origin is the center of the configured 3D box. The
preview drawing is never saved. The `image` label field points to the training
image; `raw_image` points to the original frame.

The training image inpaints the known permanent tags so their codes cannot be
used as a direct pose shortcut. Inspect the inpainted areas before training;
nearby object texture can be affected if tags overlap the object. For a stronger
test of tag-free performance, validate on a separate session without visible
tags. Split validation by capture session, camera path, or lighting rather than
adjacent frames. The collector has synthetic geometry tests; physical pose
accuracy still needs a measured box and live camera validation.
