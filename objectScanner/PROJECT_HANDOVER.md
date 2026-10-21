# Junior handover: objectScanner + objectSegmentation

Prepared 2026-10-21 (Asia/Bangkok). Start here, then read each project's guide.
This is the handover of the current research implementation. Hardware acceptance
and model generalization remain unfinished; their acceptance work is listed below.

For the step-by-step onboarding walkthrough, start with the repository's
[project handover and hands-on guide](../docs/HANDOVER.md). This document retains
the detailed two-project transfer inventory and original workstation snapshot.

## Read in this order

1. [objectScanner junior guide](docs/JUNIOR_HANDOVER.md): RGB-D collection,
   calibrated reference construction, pose training and live tracking.
2. [objectSegmentation junior guide](../objectSegmentation/docs/JUNIOR_HANDOVER.md):
   foreground masks, draft emergency-stop labels, augmentation and segmentation training.
3. Each project's [scanner session snapshot](HANDOFF.md) and
   [segmentation session snapshot](../objectSegmentation/HANDOFF.md) for verified
   state at handover; older snapshots are preserved under `.codex/handoffs/` in the original workstation roots.
4. [Local animated workflow](docs/visualizations/object-scanner/index-light.html)
   for an explanation of the scanner pipeline. Animations illustrate algorithms;
   they do not measure hardware performance.

## How the projects connect

Keep the original sibling layout:

```text
projects/
  objectScanner/
  objectSegmentation/
```

`objectSegmentation` captures RGB images with the legacy L515, prepares masks and
draft labels, and trains a single-class emergency-stop segmentation model.
`objectScanner` adds metric RGB-D geometry, calibrated eight-corner pose labels,
a reference point cloud, PnP initialization, and GICP pose tracking. Its current
camera default is the Orbbec Gemini 305; the L515 backend remains available.

Scanner collection steps 02, 05 and 06 directly use the sibling segmentation
`.venv`; step 05 also uses `backgrounds/picsum`. Scanner's r2 segmentation run
records the segmentation frozen-backbone checkpoint as its starting model in
`training/yolo/my_object_20260925_r2_onthefly_seg/args.yaml`. The resulting scanner
segmentation and pose models belong with their calibrated r2 reference and
training snapshot. Do not substitute a standalone segmentation model for the
complete pose bundle.

The scanner's legacy `weights/best.pt` is byte-identical to segmentation's
`runs/segment/emergency_stop_20260923T060602_397058Z/weights/best.pt`, confirmed by
SHA-256. It is different from `emergency_stop_frozen_backbone/weights/best.pt`.

## State the junior inherits

| Area | Verified local state | Remaining work |
| --- | --- | --- |
| Scanner r2 bundle | Snapshot integrity and frame-contract checks passed; eight corners and 3,811 reference points in `box_center` | Physical pose/frame alignment, sustained tracking and recovery acceptance |
| Default Gemini collection | Calibration/export metadata exists; no captured labels, reference, snapshot or trained model pair | A deliberate new collection and complete review/train/reference workflow |
| Scanner launch paths | Explicit r2 collection wrapper resolves its artifacts | Bare historical Phase 4 launch defaults reference missing `my_object_fresh`; use the guide's explicit session |
| Segmentation dataset | Latest version has 130 labeled records: 79 train, 50 validation, one excluded | Review automatic annotations and evaluate independent scenes/negative objects |
| Segmentation model | Frozen-backbone checkpoint and audit present | False-positive and mask-quality acceptance on fresh scenes |
| Learning material | Local two-stream visualization and deployment evidence retained | Published site was not rechecked for this handover |

## Transfer and restore

Neither project root is a Git repository. Copying only source files would lose
the datasets, calibration and trained artifacts needed to reproduce this state.
Take a filesystem backup of both complete project roots before reorganizing them.
Keep hidden `.codex/` history, relative symlinks and the nested BiRefNet checkout.
No backup copy, repository initialization, commit, upload or device migration was
performed as part of this documentation handover.

Observed disk use is approximately **17 GB scanner + 13 GB segmentation**,
including virtual environments. Allow additional destination space for a backup
and rebuilt environments. Preserve these directories in particular:

| Project | Assets to retain |
| --- | --- |
| Scanner | `datasets/`, `training/`, `references/`, `weights/`, root model files, `output/` evidence, source/scripts/tests/docs and requirement files |
| Segmentation | `input/`, `output/` including all dataset versions and `latest` symlink, `runs/`, `models/birefnet/`, `backgrounds/` including attribution/checksums, root model files, `BiRefNet/`, source/tests/docs and requirement files |

Preserve each r2 model pair together with
`datasets/my_object_20260925_r2/training_pnp/`, the production reference directory,
and the original captures/labels. Preserve the Gemini calibration history separately.
Keep crop backups and dataset exclusions so future changes can be explained.

Virtual environments and ROS build/install trees contain machine-specific paths.
Archive them as evidence if desired, but rebuild environments and ROS outputs on
a new machine using the project guides and pinned requirement files. Do not merge
the camera Python 3.10 environment with ROS/model Python 3.12 environments.
The nested BiRefNet checkout was clean at
`ebcc0bc8ec7fe919cec829f2dea656b3078acddc` when inspected.

Dataset YAML/JSON and recorded run arguments can contain `/home/jb/...` paths.
On another machine, audit active dataset paths before execution, regenerate
derived configurations through the documented tools when appropriate, and retain
original metadata as provenance. Do not edit hashed calibration snapshots just to
change a machine path. Snapshot/frame checks must pass after restoration.

Selected critical artifacts have portable, project-relative checksum lists:

```bash
# Run separately inside each project root after copying its files.
sha256sum -c docs/HANDOVER_ARTIFACTS.sha256
```

These lists identify the selected models, reference/snapshot and key metadata;
they are not a complete backup inventory or a substitute for checking the copy.
The guides explain further offline tests and the environment setup. BiRefNet
weights are present under segmentation's `models/birefnet/`; transfer that folder
so initial restoration does not depend on downloading a model again.

The published explanation is hosted from the separate `nn_ik/webui` project.
Its deployment ownership is outside these two runtime projects. Historical
publication details and rollback evidence remain in the archived scanner handoff
and `output/deploy_object_pose_20261015/`; the junior need not redeploy the site to
operate either pipeline.

## Suggested first week and acceptance

1. **Day 1 — restore and explain:** back up both roots, restore their sibling
   layout, verify checksums and environments, read both guides, and explain the
   camera/object transform and the distinction between draft masks and ground truth.
2. **Day 2 — reproduce offline:** run the guides' offline checks; inspect saved
   masks, the excluded segmentation image, r2 calibration and reference. Record
   results and any environment differences in a new dated log.
3. **Day 3 — segmentation evaluation:** after arranging the camera/session,
   collect independent positives and negatives. Review missed objects, false
   positives and mask boundaries; agree task-specific acceptance thresholds with
   the project owner before claiming generalization.
4. **Day 4 — scanner acceptance:** after hardware setup, verify axes and metric
   alignment on a stationary target, then tracking/rejection/stale/recovery
   behavior. Record conditions, observed failures and latency. A software 4 cm
   cutoff alone does not establish reliable depth at 4 cm.
5. **Day 5 — ownership review:** demonstrate restoration and both workflows,
   explain the known limitations, and agree the next milestone: completing a new
   Gemini dataset or improving fresh-scene segmentation. Save new session handoffs.

Camera operation, training and deployment are deliberate later steps; none were
started for this handover. Completion of the documentation does not close the
remaining experimental acceptance work. The recipient should record their name,
restoration location, acceptance results and next milestone when taking ownership.
