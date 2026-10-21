# Object Vision

Two related robotics vision projects, preserved together for engineering handover:

**Start with the [project handover and hands-on guide](docs/HANDOVER.md).**
It follows the myCobot280pi handover format: setup, a first-day exercise without
hardware, expected results, troubleshooting, and an ordered acceptance backlog.

| Project | Purpose | Start here |
| --- | --- | --- |
| [objectScanner](objectScanner/) | Calibrated RGB-D datasets, reference clouds and YOLO/PnP/GICP object pose tracking | [Junior guide](objectScanner/docs/JUNIOR_HANDOVER.md) |
| [objectSegmentation](objectSegmentation/) | L515 RGB capture, BiRefNet draft labels and YOLO segmentation training | [Junior guide](objectSegmentation/docs/JUNIOR_HANDOVER.md) |

Use the [combined transfer inventory](objectScanner/PROJECT_HANDOVER.md) for
artifact ownership and backup details. Keep the projects as siblings:
scanner mask generation/training uses the segmentation environment and assets.

## Clone and set up

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/jbinteam/object-vision.git
cd object-vision
# If you cloned without --recurse-submodules:
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

Recreate each environment using the project READMEs and pinned requirement files.
The environment variable above skips optional upstream Git LFS assets; it does
not omit the pinned source. Fetch those assets explicitly if resuming Isaac experiments.
Camera tooling uses Python 3.10; ROS2 Jazzy and the model environments use Python
3.12. Keep their environments separate. The scanner frontend includes its built
assets; no Node build is needed merely to use those assets.

This is a **source and documentation checkout**, not the complete experimental
workspace. Restore datasets, model weights and backgrounds separately using
[ASSETS.md](ASSETS.md) before inference, training or artifact-dependent checks.
No asset download location has been published yet. Historical absolute paths
refer to the original workstation; follow the relocation notes before running.

The available on-disk scanner baseline at handover was `my_object_20260925_r2`.
The default Gemini collection is incomplete, and bare historical Phase 4 launch
defaults reference missing assets. Use the junior guide's explicit session path
after restoring the complete calibrated bundle.

## Dependencies and validation

[DEPENDENCIES.md](DEPENDENCIES.md) lists pinned upstream submodules and the optional
CenterPose checkout/patch. Licenses and notices remain with their respective
projects/dependencies; this import does not assign a new blanket license.

Lightweight source checks and the handover's earlier offline checks are recorded
in [VALIDATION.md](VALIDATION.md). Physical tracking/frame acceptance, close-range
depth accuracy, fresh-scene segmentation and restoration on a clean machine
remain open. No hardware or training was started to publish this repository.

The original workstation directories were copied into this repository without
moving their data or environments. Treat this Git checkout as the source location
for future shared changes. Existing `HANDOFF.md` files and junior guides describe
the workstation state before Git publication; their statement that the original
roots had no Git history is historical and does not describe this repository.
