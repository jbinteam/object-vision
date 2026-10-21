# objectSegmentation — project handover snapshot

Saved: 2026-10-21T19:35:14+07:00 (Asia/Bangkok).
Project: `/home/jb/codex/projects/objectSegmentation` — not a Git repository; no branch/HEAD.

## Objective and delivery

User requested wrapping up this project and objectScanner for a junior.
Prepared [junior guide](docs/JUNIOR_HANDOVER.md) and shared
[transfer index](../objectScanner/PROJECT_HANDOVER.md), including setup references,
code map, dataset/model provenance, known limits and first-week tasks.
Documentation is complete; model acceptance and actual recipient transfer remain open.

## Changed in this session

- Added `docs/JUNIOR_HANDOVER.md` and `docs/HANDOVER_ARTIFACTS.sha256`
  (17 selected artifact/requirement checksums); linked guides from `README.md`.
- Archived previous snapshot and refreshed this one. Existing `AGENTS.md`
  already has required session-continuity behavior and was preserved.
- No application/model/data changes; no camera, training, downloads or deployment
  launched. No backup copy, repository initialization or process shutdown performed.

## Current verified state

- Pipeline: L515 RGB → BiRefNet/refinement → draft emergency-stop segmentation
  labels → custom background augmentation/training → RGB segmentation testing.
- Class 0 is red button plus yellow/black enclosure, excluding cable.
- `.venv`: Python 3.12.3, torch 2.8.0+cu128, torchvision 0.23.0+cu128,
  Ultralytics 8.3.253. `.venv-camera`: Python 3.10.18, RealSense 2.54.2.5684;
  base interpreter is under `/home/jb/anaconda3/envs/cv_py310/`.
- `output/l515_dataset/latest` resolves to `versions/20260923T060602_397058Z`:
  130 labeled records, 79 train / 50 validation / one excluded ambiguous frame.
- Continuation baseline: `runs/segment/emergency_stop_frozen_backbone/weights/best.pt`.
  Audit shows 1,365,472 frozen / 1,477,315 trainable parameters and unchanged frozen
  state. Saved CSV has 53 epochs; README records best epoch 28. Not newly trained.
- Updater/tester defaults still select the older `emergency_stop_backgrounds`
  checkpoint. Pass the intended checkpoint explicitly.
- Nested `BiRefNet/` is clean at `ebcc0bc8ec7fe919cec829f2dea656b3078acddc`;
  it is not a repository for the overall project. Local `models/birefnet/` contains
  regular config/weight files; clean-machine cold loading remains untested.
- Root size about 13 GB including environments. Preserve raw captures, full
  versioned datasets, runs, BiRefNet weights/code and backgrounds/source attribution.

## Relationship to objectScanner

Scanner mask/training wrappers use this `.venv`, BiRefNet assets and backgrounds;
keep both project roots as siblings. Scanner r2 segmentation training records the
frozen-backbone model as its initial checkpoint, then trains on scanner labels.
Scanner legacy `weights/best.pt` is byte-identical to this project's
`runs/segment/emergency_stop_20260923T060602_397058Z/weights/best.pt`, not the frozen
baseline. Do not replace scanner's calibrated model/reference/keypoint bundle.

## Checks run during this handover

- `.venv/bin/python -B -E -m pip check`: no broken requirements.
- `unittest discover -s tests -p test_polygons.py -v`: four tests passed.
- `unittest discover -s tests -p test_capture_l515.py -v`: two mock-camera tests passed.
- `sha256sum -c docs/HANDOVER_ARTIFACTS.sha256`: all 17 selected entries passed.
- Dataset symlink/counts, important paths and guide Markdown links checked.
- Historical 31-test result in `.codex/tests-freeze.log` was not rerun.
  No live camera/model accuracy evaluation or fresh-machine restoration performed.

## Limits and first next step

Automatic labels need human review; same-object/similar-scene validation does not
establish generalization. Historical filament-spool false positive around 0.51
concerns the then-used checkpoint; the frozen model needs fresh evaluation.
Keep session-based split assignments and the excluded ambiguous frame intact.
Background augmentation does not add new physical viewpoints or lighting.

Junior starts with the transfer index, backup/checksums, environment recreation
and offline checks; then proposes independent positive/negative scenes and
measurable mask/false-positive acceptance. Dataset YAML and manifests have absolute
paths; adapt a copy/config after relocation while preserving original provenance.
`download_model.py --help` is NOT a safe help probe: that script downloads directly.
Use `-E` to avoid inherited ROS Python paths. Do not launch saved jobs from this
snapshot; continue only under the current task.

Previous snapshot retained only in the original workstation project (not committed here): `.codex/handoffs/2026-10-21T19-35-14+07-00-67ce9d0d.md`.
