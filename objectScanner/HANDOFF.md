# objectScanner — project handover snapshot

Saved: 2026-10-21T19:35:14+07:00 (Asia/Bangkok).
Project: `/home/jb/codex/projects/objectScanner` — not a Git repository; no branch/HEAD.

## Objective and delivery

User requested wrapping up objectScanner and related objectSegmentation for a junior.
Prepared the documented handover; experimental hardware/model acceptance remains open.
Start with [PROJECT_HANDOVER.md](PROJECT_HANDOVER.md), then
[scanner junior guide](docs/JUNIOR_HANDOVER.md) and the linked sibling guide.
These contain setup references, architecture, artifact identities, transfer needs,
known limits and a suggested first-week plan. No recipient transfer was performed.

## Changed in this session

- Added `PROJECT_HANDOVER.md`, `docs/JUNIOR_HANDOVER.md`, and
  `docs/HANDOVER_ARTIFACTS.sha256` (20 selected artifact/requirement checksums).
- Added handover links to `README.md`; archived/replaced this snapshot.
- Prepared corresponding guide/checksums/snapshot and README links in objectSegmentation.
- No application code/model/calibration changes; no camera, training or deployment
  launched. No backup copy, commit/push, cleanup or process shutdown performed.
- `AGENTS.md` already contains the required startup/handoff behavior; left unchanged.

## Verified current state

- Available runtime bundle: `my_object_20260925_r2`, 212 labels, paired split
  186 train / 26 validation, calibrated `box_center` reference with 3,811 points.
- Models: `training/yolo/my_object_20260925_r2_onthefly_seg/weights/best.pt`
  and `training/yolo/my_object_20260925_r2_pose_refined_pose/weights/best.pt`.
- Preserve `datasets/my_object_20260925_r2/training_pnp/` and
  `datasets/my_object_20260925_r2/references/my_object/` with those weights.
- Default `my_object_gemini305` session has calibration/export metadata/history,
  zero labels and no reference/training snapshot/model pair. Not ready for live use.
- Bare Phase 4 launch defaults reference absent `my_object_fresh` assets. The
  explicit-session collection wrapper is the documented path for the r2 bundle.
- Legacy `weights/best.pt` equals sibling segmentation's
  `emergency_stop_20260923T060602_397058Z` checkpoint by SHA-256; it differs from
  the newer frozen-backbone checkpoint used to initialize scanner's r2 training.
- Scanner mask/training workflows use sibling objectSegmentation `.venv`,
  BiRefNet assets and backgrounds. Preserve sibling layout and separate environments.
- Observed total size about 17 GB scanner plus 13 GB segmentation, including venvs.
  Absolute paths and interpreter dependencies require reviewed relocation/rebuild.

## Checks run during this handover

- `scripts/pnp_snapshot.py verify datasets/my_object_20260925_r2`: passed.
- `scripts/verify_phase4_frame_contract.py` with frozen r2 calibration/keypoints:
  passed, eight corners and 3,811 points in `box_center`.
- `scripts/train_yolo_seg_pose.py --dataset
  datasets/my_object_20260925_r2/yolo_seg_pose_onthefly --task both --dry-run`:
  passed, paired split 186/26 for both tasks; no training performed.
- `sha256sum -c docs/HANDOVER_ARTIFACTS.sha256`: all 20 passed; sibling all 17 passed.
- Local Markdown links in combined index/both guides: 19 checked, zero missing.
- Full regression suites, hardware, fresh-machine restoration and public website
  were not tested. Older reports remain historical evidence, not fresh results.

## Contracts and unfinished acceptance

Preserve metres, ordered eight corners, fixed `box_center`, and
`P_camera = T_camera_object @ P_object`. Keep calibrated PnP snapshot immutable.
PnP initializes/recovers tracking; GICP seeds from the last accepted RAW pose;
filter only accepted poses. Stale returns do not count as ordinary rejections.

Open work: physical frame alignment, sustained tracking/recovery/latency,
long-duration GPU behavior, and measured close-range accuracy. The 0.04 m software
cutoff does not establish 4 cm depth accuracy. New Gemini training needs a new
complete collection/review/reference/train cycle; do not alter the r2 baseline.

## Earlier publication context

Previous session completed the illustrative two-stream processing visualization
and Arm Lab publication. Local figure: `docs/visualizations/object-scanner/`.
Historical live URL: https://arm.gradual-lab.dev/research/object-pose
Separate site source: `/home/jb/codex/projects/nn_ik/webui`.
Deployment evidence: `output/deploy_object_pose_20261015/`. Publication, rollback
paths and earlier test details are in the archived snapshot, not reverified today.

## First next step

Junior reads the transfer index, backs up both complete roots, verifies the selected
hashes and environments, then reproduces offline checks before arranging hardware
acceptance. Documentation handover is complete; delivery to a named person and
new-machine restoration are not verified. Follow the current user's next task;
this snapshot does not authorize launching saved commands.

Previous snapshot retained only in the original workstation project (not committed here): `.codex/handoffs/2026-10-21T19-35-14+07-00-ae76dde3.md`.
