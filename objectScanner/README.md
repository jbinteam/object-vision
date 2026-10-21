# RGB-D Object Scanner (Orbbec Gemini 305)

Taking over this project? Start with the [hands-on handover](../docs/HANDOVER.md),
then the [two-project transfer inventory](PROJECT_HANDOVER.md) and
[scanner junior guide](docs/JUNIOR_HANDOVER.md). They identify the verified artifact
bundle, setup steps and remaining acceptance work.

For RGB image to object-center pose dataset capture with a tagged turntable,
see [the turntable dataset guide](docs/TURNTABLE_POSE_DATASET.md).
For the current step-by-step collection and calibrated RGB-D TSDF reference
workflow, see [the collection scripts](scripts/collect_process/README.md).
For animated processing steps and deep dives into dataset collection, training, and pose estimation,
open [the two-stream visualization](docs/visualizations/object-scanner/index.html)
in a browser.

Scan a stationary object by moving an Orbbec Gemini 305 around it. Acquisition
uses Orbbec SDK v2 through the Python wrapper; Open3D provides tracking and TSDF
fusion. The optional RealSense L515 backend remains available with
`--camera-backend realsense`. The default result is a **colored 3D point cloud in metres**.

See [Gemini 305 installation and migration](docs/GEMINI305_CAMERA.md) for SDK,
USB permissions, device selection, and the calibration workflow.

## ROS2 object cloud (Phase 1)

RGB-D capture, YOLO segmentation, and synchronized object-cloud extraction are
implemented for ROS2 Jazzy. From this directory:

```bash
./scripts/build_phase1.sh
./scripts/run_phase1.sh rviz:=true
```

Uses `weights/best.pt` on CPU and Gemini 305 in the isolated camera environment.
Select the legacy camera with `camera_backend:=realsense`. See
[Phase 1 setup, parameters, verification, and limitations](docs/PHASE1_RGBD_SEGMENTATION.md).
The earlier L515 hardware test verified streaming and empty-mask behavior.
Gemini acquisition and worker-pipe smoke checks pass; actual ROS camera-topic
streaming and live target cloud/RViz acceptance are pending. Fixed-reference ICP
and object pose TF are Phase 2.

## ROS2 fixed-reference ICP (Phase 2)

Estimate the reference object's pose and show its XYZ frame and aligned reference
in RViz:

```bash
./scripts/build_phase1.sh
./scripts/run_phase2.sh
```

See [Phase 2 configuration, frame display, and validation](docs/PHASE2_FIXED_REFERENCE_ICP.md).
The reference is loaded once; each observation is independently centroid-seeded.
Previous-pose tracking is Phase 3. Synthetic pose/TF/frame tests pass; physical
frame alignment still requires a detected object and inspection in RViz.

## ROS2 previous-pose tracking (Phase 3)

```bash
./scripts/build_phase1.sh
./scripts/run_phase3.sh
```

Uses the last accepted raw ICP pose to seed registration of the same fixed
reference against each new observed cloud. Rejected candidates cannot replace the
seed; expired seeds return to centroid initialization. See
[Phase 3 behavior, configuration, tests, and timings](docs/PHASE3_PREVIOUS_POSE_TRACKING.md).

CUDA YOLO and ICP are available through a separate environment and launcher:

```bash
./scripts/build_cuda.sh
./scripts/run_phase3_cuda.sh
```

See [CUDA setup, measured performance, and hybrid GPU/CPU preset](docs/CUDA_TRACKING.md).
The revised CUDA point-to-plane loop is faster than cached CPU ICP on the
supplied reference in paired tests; the hybrid preset remains available. Stop the current camera launch before switching versions.

A proposed successor architecture adds YOLO keypoints and PnP initialization
before coarse-to-fine GICP. Its preserved implementation brief is in
[YOLO Pose/Segmentation → PnP → GICP](docs/YOLO_PNP_GICP_IMPLEMENTATION_PROMPT.md).
The runnable implementation is in
[Phase 4: YOLO + PnP + GICP](docs/PHASE4_YOLO_PNP_GICP.md).
See the [current performance review and reproducible cloud benchmark](docs/PHASE4_PERFORMANCE.md).

## Web UI: crop and assign obj_frame

```bash
source .venv/bin/activate
python webui.py
```

Open **http://127.0.0.1:8080**. Use `--port 8081` if port 8080 is occupied.
This is a local application; uploads stay on this machine. Cloud processing uses
CPU only. The browser uses WebGL for its limited point-cloud preview, requests
the low-power graphics adapter, and renders only when the view changes.

The turntable collector can open this editor for a diagnostic moving-camera
scan with **Inspect scan [E]**. It supplies session-specific `--scan-dir`,
`--reference-dir`, `--object-frame box_center`, and `--fixed-frame-from-scan`;
the scan sidecar's AprilTag-derived `T_scan_object` keeps that diagnostic export
in the capture frame. The production Phase 4 reference is now built directly
from the saved calibrated RGB-D captures by `04_build_reference.sh`.

1. **Source:** choose a scan from `output/`, or open a PLY from your computer.
   Choose its source units before loading (scanner exports use metres).
2. **Crop:** adjust the X/Y/Z minimum and maximum bounds using sliders or
   numeric inputs. Excluded points appear dark and can be hidden. The crop is
   box that can be rotated: select **Rotate box** and drag its colored rings,
   or enter roll/pitch/yaw under **Box rotation**. Min/max move individual
   faces along the box's local X/Y/Z axes, even after rotation; the opposite
   face stays fixed. Rotation uses Rz · Ry · Rx about the current box center.
   **Esc** cancels a drag, **Zero rotation** removes the rotation,
   and **Reset** restores the full source bounds and zero rotation. Preview,
   point picking, full-resolution counts and export all use the rotated box.
3. **Frame:** choose **Crop center**, enter origin/rotation, or pick three points:
   the origin, a point along +X, and a point toward +Y. +Y is orthogonalized
   against +X; +Z is their cross product. Orbit to a useful view before picking.
   Use **Move** to drag the colored axis arrows, plane squares, or center handle.
   Use **Rotate** to drag the rotation rings around the frame's local axes.
   Numeric fields and the export transform update immediately. **Esc** cancels
   the current drag; **Off** hides the handles. Drag elsewhere to orbit.
   Handles appear in the **Frame** tab in **Scan frame** view; selecting Move or
   Rotate switches back to that view when needed.
   Switch to **Object frame** to inspect the result. Editing never changes the
   original scan file.
4. **Export:** name the reference, optionally choose a voxel size in mm, and
   export. A zero voxel size retains every cropped source point. Normals can be
   re-estimated for point-to-plane ICP. A download bundle is also available.

Exports are saved under `references/<name>/`; existing names are not overwritten:

| File | Meaning |
| --- | --- |
| `reference.ply` | Cropped cloud transformed into `obj_frame`, always in metres |
| `frame.json` | Source hash, crop, origin/rotation, units and both 4×4 transforms |
| `README.txt` | Coordinate and ICP usage notes |
| `reference_bundle.zip` | PLY, JSON and usage notes in one download |

Matrices are row-major JSON arrays and act on homogeneous **column vectors**:

```text
p_scan = T_scan_obj @ p_obj
p_obj  = T_obj_scan @ p_scan
R_scan_obj = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

For later ICP, use `reference.ply` as the source and the live scene as the target.
The returned source-to-target registration transform is then `T_scene_obj`.
Provide an adequate initial pose; the editor does not implement global matching
or guarantee ICP convergence. Re-estimated normals face the original scan origin;
inspect their orientation on objects scanned from several viewpoints.

The display samples at most 120,000 points, but cropping and export use the full
cloud, up to 5 million points. Uploads are limited to 256 MB. Loaded cloud sessions
are temporary (the two most recent clouds are retained); exported references are
durable files. The page requires a WebGL2-capable browser.

The compiled web assets are included. After modifying the frontend:

```bash
cd webui
npm ci
npm run build
npm test
```

## Run on this machine

A Python 3.10 virtual environment with the dependencies is installed in `.venv`.

```bash
cd /home/jb/codex/projects/objectScanner
source .venv/bin/activate
python scanner.py --list-devices
python scanner.py --safe --seconds 30
```

After a reported GPU freeze, reconstruction now defaults to CPU. `--safe`
additionally disables both preview windows, downsamples by at least 2, and uses
voxels of at least 5 mm. This avoids the scanner's CUDA and OpenGL rendering paths.
Run without `--safe` for the color/depth preview. The live 3D preview is off by
default (`--no-3d`); add `--3d` to enable it. `--safe` and `--no-view` disable both
windows even when `--3d` is passed.
Keep the object about 0.25–1.5 m from the camera. Move the camera slowly around
the stationary object; leave some surrounding geometry visible for tracking.
Do not rotate the object on a turntable while the background stays still.
Tracking fitness defaults to 0.1; increase `--min-fitness` for stricter frame
rejection if the scene provides sufficient overlapping geometry.

- **Q / Esc**, or closing either preview: stop and save.
- **Ctrl+C**: stop and save in headless mode too.
- **D** in the RGB window: toggle the valid-depth overlay.
- **Mouse** in the 3D window: rotate, pan, zoom; **R** resets the view.
- **C** in the 3D window: switch between original colors and grey.

Example scans:

```bash
# Name the scan: saves output/mug.ply (the .ply extension is optional)
python scanner.py --name mug --seconds 30

# Enable the optional live 3D preview
python scanner.py --name mug_3d --3d

# Save a named point cloud; also extract a mesh
python scanner.py -s 40 -o output/my_object.ply --mesh

# Try removing the support table and keeping the largest cluster
python scanner.py -s 40 -o output/isolated.ply --isolate

# Lower processing cost on CPU
python scanner.py --device cpu --downscale 2 --voxel 5 --no-3d

# Headless capture
python scanner.py -s 30 --no-view

# Inspect a saved point cloud
python viewer.py output/my_object.ply
```

Use `python scanner.py --help` for all options. Existing output files are
protected from overwriting. Use `-n/--name` for a filename in the project's
`output/` folder, or `-o/--output` for an explicit PLY path; the options cannot be
combined. Without either option, scans get timestamped filenames under `output/`.

## Outputs

| File | Contents |
| --- | --- |
| `scan_TIMESTAMP.ply` | Fused colored point cloud |
| `scan_TIMESTAMP_run.log` | Effective device/settings, processing stages and Python errors |
| `scan_TIMESTAMP_scan.json` | Camera identity, settings, processed-image intrinsics, accepted camera poses and counts |
| `scan_TIMESTAMP_trajectory.ply` | Camera centers as orange points |
| `scan_TIMESTAMP_mesh.ply` | Optional triangle mesh with `--mesh` |

Coordinates are metres, with the first accepted camera view defining the world
origin: X right, Y down, Z forward. JSON poses map camera coordinates into world
coordinates. RGB and depth are aligned, rectified, and resized together; sensor
depth units are converted using the selected camera's reported scale.

## Fresh installation

Use **Python 3.10 on Linux x86-64** for the existing scanner environment.
The Orbbec Python wrapper also provides Python 3.12 wheels, while the optional
pinned L515 wheel requires Python 3.10/3.11. On this machine Python 3.10 is available
at `/home/jb/.pyenv/versions/3.10.20/bin/python`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-orbbec.txt
```

For the optional L515 backend, also install `requirements-realsense.txt` and use
`--camera-backend realsense` (ROS: `camera_backend:=realsense`).

The L515 is retired hardware. RealSense lists SDK **2.50.0 as the last validated
release**, and **2.54.2 as supporting L515 without validation**. This project pins
`pyrealsense2==2.54.2.5684` to retain L515 support with Python 3.10/3.11 wheels;
do not blindly upgrade it. See the official
[release notes](https://github.com/realsenseai/librealsense/wiki/Release-Notes)
and [alignment example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py).

Connect through USB 3. If the camera is not visible, check `lsusb`, close other
camera applications, and install the selected SDK's udev rules (see the
[Gemini 305 guide](docs/GEMINI305_CAMERA.md)). Startup errors are reported without changing firmware or camera presets.
CPU is the default. `--device cuda` explicitly selects CUDA; `--device auto` opts into CUDA when available. Avoid those modes while investigating the reported GPU freeze. A desktop display
is needed for previews; use `--no-view` without one.

## Limits and testing

Tracking uses depth geometry and aligns each frame to the fused model. There is
no global loop closure or automatic relocalization; a full orbit may leave a
seam. If tracking is lost, return toward the last successful view. Flat, shiny,
transparent, dark, or very small objects can provide insufficient depth geometry.
The first frame must contain enough valid depth to initialize the model.

Isolation is a heuristic (dominant plane removal plus largest cluster); inspect
the output since a flat object or clutter can confuse it. The optional mesh is
not guaranteed watertight. The 3D preview updates every 15 accepted frames by
default; `--preview-every` changes the frequency.

```bash
python -m unittest discover -s tests -v
```

Tests cover depth scale/range/ROI, invalid settings, overwrite protection, empty
initial frames, object clustering and synthetic reconstruction/export. Hardware
capture requires a supported connected RGB-D camera.

See `NOTICE` and `LICENSE` for attribution to the Orbbec reference.

## GPU freeze investigation (2026-09-22)

The previous boot recorded NVIDIA Xid 79 (GPU has fallen off the bus) and Xid 154
(GPU reset required) at 13:23:06, followed by display-driver progress errors.
These establish a GPU-level failure, but do not identify the root cause or prove
VRAM exhaustion. See [NVIDIA Xid documentation](https://docs.nvidia.com/deploy/xid-errors/).
The GPU responded to a status query after reboot; no GPU workload was rerun as
part of this investigation.

Only Open3D's known singular-system tracking error is treated as a lost frame.
Unexpected reconstruction errors now stop capture and skip model extraction and
export, avoiding further model operations on a potentially faulted CUDA context.
The current in-memory scan cannot be recovered through that path. Camera-read
errors can still save a healthy reconstructed model. Each scan writes a stage log;
Python's fault handler also reports supported native faults to stderr.

These are containment measures, not a verified fix for the driver/GPU failure.
Python error handling cannot recover a GPU that has stopped responding. Use
`python scanner.py --safe --seconds 10` while diagnosing it. No driver, firmware,
or system configuration changes were made.

## Native-call stalls

A subsequent CPU run stalled while processing the third frame. A synthetic
20-frame CPU reconstruction completed, so the underlying stall is not yet
reproduced or diagnosed. The CLI now runs capture/reconstruction in a spawned
worker supervised by a separate process. This lets the watchdog terminate a
stuck native operation even when Python cannot handle Ctrl+C in that worker.

`--stall-timeout 15` is the default inactivity limit per processing operation;
initial library imports get at least 30 seconds. The watchdog covers camera
reads, reconstruction, preview, cleanup and saving. `--seconds` is the capture
duration, not a total deadline including startup and saving. Large legitimate
exports may need a longer stall timeout.

Ctrl+C requests a graceful stop, with three seconds to save and exit. Press it
again to force termination immediately. If the worker is stuck, its in-memory
scan is lost. `_run.log` now names tracking, integration and raycasting stages;
`_native.log` receives a Python stack dump on a watchdog stop on Linux. A native
driver/kernel hang can still prevent OS-level process termination; this is not
a recovery mechanism for the earlier GPU bus failure.

`--safe` also limits numerical-library thread pools to one thread. Tracking
rejects invalid rigid poses or jumps exceeding 25 cm / 45 degrees per accepted
frame before integration. These are conservative containment measures, not a
confirmed explanation for the stall.

## USB stream selection

Gemini 305 prefers an advertised 30 FPS color/depth pair and aligns depth to
rectified color pixels. See the [Gemini guide](docs/GEMINI305_CAMERA.md) for its
format, geometry, and USB requirements. The following historical profile and
warm-up behavior applies to the optional L515 backend.

Startup queries the camera's advertised profiles and verifies that the depth and
color pair can be resolved together. It prefers 640x480 depth at 30 FPS on USB 3,
and falls back to the advertised 320x240 depth mode on USB 2. Color uses the
smallest compatible RGB8 mode at 30 FPS. USB speed and chosen modes appear in the
terminal, run log and saved camera metadata. USB 2 has lower depth sampling;
software upsampling after color alignment does not restore that detail.

Close other camera applications before starting a scan, including OrbbecViewer
for Gemini or RealSense Viewer for L515. Viewer executables are separate from
the Python environment, so activating `.venv` does not change their SDK versions.

Camera warm-up now waits for 15 complete RGB-D pairs within a total 10-second
budget, retrying brief waits without frames. It reports the received pair count
on failure. The watchdog remains active, so a `--stall-timeout` shorter than this
budget can terminate startup sooner. USB enumeration and resolvable profiles do
not by themselves guarantee that frames will arrive. A bounded camera-only test
on 2026-09-22 received RGB-D frames using 640x480 depth / 960x540 color after a
reported warm-up timeout; the intermittent delivery issue remains undiagnosed.
