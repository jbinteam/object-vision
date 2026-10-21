# Collection process

Run one script at a time from the project root (or use `./01_capture.sh`, etc.
from this folder). With no arguments, all scripts use the Orbbec
`my_object_gemini305` session and `my_object` category. Each also accepts
`[SESSION] [CATEGORY]`; a different session uses its own name as the default
category. `COLLECT_SESSION` and `COLLECT_CATEGORY` can set these defaults in the
environment. Additional arguments are passed to the underlying Python command
(or ROS launch in step 7). Nothing starts automatically.

Gemini 305 is the default camera. Steps 1 and 7 explicitly select the `orbbec`
backend; `COLLECT_CAMERA_BACKEND` can override it for both. The existing L515
capture calibration cannot be resumed with Gemini, so the default session is
separate from `my_object_20260925_r2`. For another object or a changed setup,
choose a new session and use it for every step, for example
`export COLLECT_SESSION=widget_gemini305 COLLECT_CATEGORY=widget`.
See [Gemini 305 migration](../../docs/GEMINI305_CAMERA.md) for SDK installation.

```bash
scripts/collect_process/01_capture.sh
scripts/collect_process/02_generate_masks.sh
scripts/collect_process/03_export_yolo.sh
scripts/collect_process/04_build_reference.sh
scripts/collect_process/05_train_segmentation.sh
scripts/collect_process/06_train_pose.sh
scripts/collect_process/07_run_live.sh
```

To select a specific camera, run
`scripts/collect_process/01_capture.sh my_object_gemini305 my_object --serial SERIAL`;
pass `camera_serial:=SERIAL` after the session/category in step 7.
The original dataset and pinned models/reference remain available explicitly:
`scripts/collect_process/07_run_live.sh my_object_20260925_r2 my_object` uses those
assets with the Orbbec camera. Add `camera_backend:=realsense` to use the L515.
To resume capture with the original L515 and unchanged tag setup, use
`scripts/collect_process/01_capture.sh my_object_20260925_r2 my_object --camera-backend realsense`.

Step 1 is interactive: set the box, remove setup tags, and capture RGB-D views.
Keep the object and table tags fixed after calibration. `P` optionally records a
moving-camera TSDF scan for diagnostics; that scan is not the production GICP
reference. It defaults to CUDA, full RGB-D resolution, and a 60-second stall
watchdog. `01b_edit_reference.sh SESSION CATEGORY` reopens its diagnostic crop
UI without starting the camera.

Orbbec collection keeps depth from **0.04–1.5 m**, including the optional `P`
scan; use `--depth-min` and `--depth-max` after the session/category to change
these software cutoffs. The `realsense` and `auto` selections retain a 0.25 m
default near cutoff. Each saved label records its range and frame intrinsics;
step 4 uses the saved aligned depth. The 4 cm cutoff does not establish measured
depth accuracy at that distance. Keep at least two mapped permanent tags visible
in every captured view.

Step 2 writes mask previews; inspect every image in
`datasets/SESSION/centerpose_masks/review/` before continuing. Step 3 exports
paired segmentation and eight-keypoint labels. Step 4 fuses the masked RGB-D
captures using each saved `T_camera_object` into a TSDF reference directly in
the calibrated `box_center` frame. It crops automatically to the metric box
plus 3 mm and needs no manual axis or crop adjustment. It refuses to replace an
existing reference directory; archive an older export before rerunning it.
`04_align_reference.sh` remains as an alias for this new step.
If the fused PLY has isolated defect points, run `04b_crop_reference.sh` to
crop only its geometry in the browser. It locks the calibrated frame, saves a
timestamped backup of the original bundle, and updates the reference point
count and bundle metadata when you save. Review the retained point count
before saving; the crop keeps points inside its box.

Steps 5 and 6 train the separate YOLO models. Before either training starts,
the scripts save `calibration.json` and `object_keypoints.yaml` under
`datasets/SESSION/training_pnp/`, with hashes in `manifest.json`. An existing
snapshot is reused only when the current inputs still match; a changed box
requires a new session. Step 7 launches the live camera and tracking using
the saved PnP keypoints. It checks the snapshot hashes and verifies that its
eight corners and the production reference share `box_center` before starting
ROS. Reconfiguring the collection later cannot change the live PnP geometry.
Step 7 publishes the camera image with the projected 3D wireframe at
`/object_pose/box_image`. Its default camera publication cap is 20 Hz; pass
`camera_rate:=15` after the category to choose another cap. Camera capture
prefers 30 FPS; output rate depends on inference and tracking. Pass
`camera_backend:=orbbec camera_serial:=SERIAL` to select Gemini explicitly, or
`camera_backend:=realsense` for the legacy L515.

Step 6 starts from the previously successful pose checkpoint. To use another
checkpoint, set `POSE_START_WEIGHTS=/absolute/path/to/best.pt` for that command.
Training commands create new run directories by default, so choose a new
session name for a new collection. For a headless live run, append
`rviz:=false` to step 7.

The capture script uses the current 14 mm setup tags, 28 mm permanent tags,
and `tagStandard41h12` IDs. Pass different collector options after the category
when your printed tags differ. See `docs/TURNTABLE_POSE_DATASET.md` for the
physical layout and controls.
