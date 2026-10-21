# Phase 4: YOLO segmentation + 8 keypoints → PnP → GICP

The canonical convention throughout this phase is:

```text
P_camera = T_camera_object * P_object
```

`reference.ply`, `object_keypoints.yaml`, PnP, GICP, and the final TF all use
the calibrated `box_center` frame. Do not invert or re-center the generated
reference at runtime.

For new collections, `scripts/collect_process/04_build_reference.sh` integrates
the reviewed masked depth views using their saved `T_camera_object` poses. The
production TSDF point cloud is therefore generated directly in `box_center`,
the same frame as the eight PnP corners. The moving-camera Scan PLY is retained
only for diagnostics.

## Train

The paired export contains one file per captured source view plus its foreground
mask. Backgrounds are sampled on the fly for both tasks, so no `__bg_*` training
PNGs are required. Validation remains unchanged. Horizontal geometric flipping
is disabled because it would change the semantic corner ordering.

```bash
../objectSegmentation/.venv/bin/python -E scripts/train_yolo_seg_pose.py \
  --dataset datasets/my_object_fresh/yolo_seg_pose_onthefly \
  --project training/yolo \
  --name my_object_fresh --task both --epoch-repeats 12 \
  --backgrounds ../objectSegmentation/backgrounds/picsum
```

Outputs:

```text
training/yolo/my_object_fresh_seg/weights/best.pt
training/yolo/my_object_fresh_pose/weights/best.pt
```

`--epoch-repeats 12` replaces the training volume formerly supplied by twelve
materialized copies. Each repeated read chooses a fresh background, while the
21 source images and the validation set remain unmodified.

For the current small capture set, the deployed pose checkpoint is the fully
unfrozen refinement pass:

```bash
../objectSegmentation/.venv/bin/python -E scripts/train_yolo_seg_pose.py \
  --task pose \
  --dataset datasets/my_object_fresh/yolo_seg_pose_onthefly \
  --pose-model training/yolo/my_object_rgbd_unfrozen_pose/weights/best.pt \
  --project training/yolo --name my_object_fresh_pose_refined \
  --epochs 100 --patience 25 --freeze 0 --lr0 0.00005 --epoch-repeats 12
```

Its output is
`training/yolo/my_object_fresh_pose_refined_pose/weights/best.pt` and Phase 4 uses
that checkpoint by default.

The older `my_object_fresh` reference was cropped from a moving-camera scan.
For that historical asset, residual scan translation was aligned against the
calibrated capture depths without redefining `box_center`:

```bash
.venv-ros-cuda/bin/python scripts/align_reference_to_capture.py \
  --dataset datasets/my_object_fresh \
  --reference datasets/my_object_fresh/references/my_object/reference.ply
```

The default Phase 4 launcher still uses that historical
`reference_aligned.ply`. The collection script's live step overrides the path
with the current session's calibrated `reference.ply`.

The eight PnP corners come from the capture calibration's metric box. Before
launching the collection script's live step, check the current asset contract
with:

```bash
.venv-ros-cuda/bin/python scripts/verify_phase4_frame_contract.py \
  datasets/my_object_20260925 my_object
```

## Build and run

```bash
scripts/build_phase1.sh
scripts/run_phase4.sh
```

Use the installed CUDA environment for GPU YOLO inference:

```bash
scripts/build_cuda.sh
scripts/run_phase4_cuda.sh
```

For external already-aligned RGB-D topics:

```bash
scripts/run_phase4.sh camera:=false
```

The default `.venv-ros` installation runs both YOLO models on CPU. A CUDA ROS
environment can select `config/devices_cuda.yaml` after CUDA PyTorch and the
ROS dependencies are installed together.

Main outputs:

- `/object_pose/pnp`, `/object_pose/gicp`, `/object_pose/filtered`
- `/object_pose/base` when `base_link → camera_color_optical_frame` is available
- compatibility output `/object/pose`
- TF frames `box_center`, plus optional `object_pnp`, `object_gicp`, and
  `object_filtered`
- `/object_pose/tracking_state` (`INITIALIZATION`, `TRACKING`, or `LOST`)
- `/object_pose/detection_image` (`sensor_msgs/Image`, `rgb8`): color frames
  with the selected segmentation bounding box and confidence. The original
  RGB frame is published when there is no detection.
- `/object_pose/box_image` (`sensor_msgs/Image`, `rgb8`): camera frames with the
  calibrated eight-corner 3D box projected from the accepted filtered GICP pose.
  Rejected frames show `NO ACCEPTED POSE` instead of an old box.
- `/object_pose/pnp_reprojection`, `/object/reference_cloud_pnp`, and
  `/object/reference_cloud_gicp`

PnP is used for startup/reinitialization. After one accepted GICP result, the
previous accepted unfiltered result initializes the next GICP fit. Rejected
poses never enter the temporal filter.

## Independent geometry tests

PnP observation JSON accepts ordered arrays or objects with `id`, `u`, `v`,
`confidence`, and `valid`, plus `camera_matrix`/`K`:

```bash
.venv/bin/python scripts/offline_pnp_test.py \
  --object-keypoints datasets/my_object_fresh/yolo_seg_pose_onthefly/object_keypoints.yaml \
  --observation observation.json --image frame.png --overlay /tmp/pnp.png
```

GICP accepts a reference, observed PLY/PCD, and JSON initial transform:

```bash
.venv-ros/bin/python scripts/offline_gicp_test.py \
  --reference datasets/my_object_fresh/references/my_object/reference_aligned.ply \
  --observed observed.ply --initial T_camera_object.json --visualize
```

All quality gates and coarse/medium/fine settings are in `config/phase4.yaml`.
Diagnostics report PnP inliers/reprojection error, GICP fitness/RMSE/overlap,
per-level runtime, temporal jumps, initializer selection, and total runtime.
