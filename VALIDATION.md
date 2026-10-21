# Publication validation — 2026-10-21

Checks below used the prepared source checkout at
`/home/jb/codex/projects/object-vision`, with existing original-workspace Python
environments. This verifies the packaged source, not a clean-machine installation.

## Checks on this checkout

- Python AST parsing: 121 project Python files passed, excluding upstream submodules.
- Scanner selected source/docs/frontend/package comparison: 246 files matched
  the original workspace; generated pytest caches were omitted.
- Scanner offline unit tests: **62 passed**, using mocks/synthetic fixtures and
  temporary exports. No camera was opened.
- Segmentation tests: **4 polygon + 2 mocked capture tests passed**.
- Visualization: `node docs/visualizations/object-scanner/validate-content.mjs`
  passed **12 lessons and 53 source citations** from the scanner directory.
- Frontend: **8 geometry tests passed** using the staged source and a transient
  Node resolver for the original installed `three@0.180.0`. Plain Node testing
  initially failed because the fresh checkout had no `node_modules`; `npm ci`
  and ordinary `npm test` were not run during publication. To reproduce normally,
  run `npm ci` then `npm test` inside `objectScanner/webui/`.
- All three submodule index pins match `DEPENDENCIES.md`. Optional upstream
  LFS smudging was stopped during setup; pinned source checkout completed with
  `GIT_LFS_SKIP_SMUDGE=1`. No LFS payload was added to this repository.
- Selected source/docs review found no credential filenames, private-key patterns
  or high-confidence tokens. This was a pattern review, not a formal security audit.
- Inventory excludes virtual environments, caches, model binaries, raw datasets
  and credentials. Three's full MIT license accompanies the bundled frontend;
  existing project and visualization licenses/notices are retained.

Scanner test command used, from the staged `objectScanner/` directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/jb/codex/projects/objectScanner/.venv/bin/python -B - <<'PY'
import unittest
patterns = [
    'test_camera_backends.py', 'test_camera_profiles.py', 'test_orbbec_camera.py',
    'test_turntable_dataset.py', 'test_generate_yolo_seg_pose_dataset.py',
]
suite = unittest.TestSuite(
    unittest.defaultTestLoader.discover('tests', pattern=p) for p in patterns
)
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

Segmentation commands used, from the staged `objectSegmentation/` directory:

```bash
/home/jb/codex/projects/objectSegmentation/.venv/bin/python -B -E -m unittest discover -s tests -p test_polygons.py -v
/home/jb/codex/projects/objectSegmentation/.venv/bin/python -B -E -m unittest discover -s tests -p test_capture_l515.py -v
```

Use the corresponding recreated environment paths on another machine.
Imported historical logs, patch context and bundled shader strings retain their
original whitespace; the import was not reformatted to remove historical
`git diff --check` warnings.

## Earlier handover checks on original complete workspaces

On the same day, before preparing this source checkout: all 37 selected artifact
checksums passed; scanner r2 PnP snapshot/frame verification and paired-data dry run
passed (eight corners, 3,811 reference points, 186/26 train/validation split);
segmentation `pip check` was clean. See the project handoffs for details.
Those artifact-dependent checks require restoring the files described in ASSETS.md.

## Not validated here

No full regression suite, hardware capture, training, deployment, fresh-scene
model evaluation or clean-machine restoration was performed. Physical frame
alignment, sustained Gemini tracking/recovery and close-range depth accuracy
remain open acceptance work. This publication does not change model/runtime code.
