# Dataset and model transfer

The Git repository carries source, tests, documentation, frontend assets and
selected provenance metadata. Large experimental assets remain on the original
workstation; there is currently **no hosted download or release asset bundle**.
Arrange a separate copy from the project owner before attempting live operation.

| Source on original workstation | Destination relative to this checkout |
| --- | --- |
| `/home/jb/codex/projects/objectScanner/` | `objectScanner/` |
| `/home/jb/codex/projects/objectSegmentation/` | `objectSegmentation/` |

Use the transfer inventories in the [combined handover](objectScanner/PROJECT_HANDOVER.md)
and both junior guides. Preserve symlinks and the complete dataset version history.
The original roots total approximately 30 GB including their environments.

## Required scanner baseline

Restore `objectScanner/datasets/my_object_20260925_r2/`, its paired runs
`training/yolo/my_object_20260925_r2_onthefly_seg/` and
`training/yolo/my_object_20260925_r2_pose_refined_pose/`, and the documented source
checkpoints. Keep calibration, `training_pnp/`, reference geometry and metadata
together with their model pair. Keep `datasets/my_object_gemini305/` calibration
history separately; it has no completed training/reference bundle.

Legacy scanner paths additionally use `weights/best.pt`, root YOLO checkpoints
and `references/`. Preserve `output/` and historical `training/` as evidence
when transferring the full research archive. Restore needed historical datasets
before running any corresponding older examples.

## Required segmentation assets

Restore `input/`, `output/` including `l515_dataset/versions/` and its `latest`
symlink, `models/birefnet/`, `runs/`, `backgrounds/picsum/` including `sources.json`,
and root YOLO checkpoints. Scanner mask generation depends on BiRefNet code/weights;
scanner segmentation training also depends on this background library and Python
environment. Do not assume Git submodules contain downloaded model weights.

## Provenance and verification

Each project's `docs/provenance/` contains small metadata copies in mirrored
source paths. These describe the existing artifacts; they do not provide images,
point clouds or weights. Three scanner metadata files are also retained at their
original paths because the workflow visualization cites them. Their presence does
not mean the surrounding dataset/training directory is complete.

After restoring assets, run inside **each** project directory:

```bash
sha256sum -c docs/HANDOVER_ARTIFACTS.sha256
```

The manifests cover 20 scanner and 17 segmentation files, including selected
weights, calibration/reference metadata and requirement pins. A fresh source-only
checkout is expected to report missing files. Do not suppress those failures and
call the dataset restored. These manifests are not complete archive checksums.

Rebuild virtual environments and ROS build/install outputs on the new machine;
copying them is not a portable setup. Paths in YAML/JSON may point to the original
workstation. Preserve original records, adapt derived dataset configs in a copy,
and rerun snapshot/frame-contract checks. Do not rewrite the immutable calibration
snapshot merely to relocate files. Clean-machine restoration is not yet verified.
