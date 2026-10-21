// Source of truth: scripts/collect_process/{01_capture,02_generate_masks,
// 03_export_yolo,04_build_reference,05_train_segmentation,06_train_pose,
// 07_run_live}.sh and README.md; scripts/build_calibrated_reference.py;
// scripts/generate_yolo_seg_pose_dataset.py; scripts/pnp_snapshot.py.
// Runtime: ros2_ws/src/object_pose_tracking/config/phase4.yaml and
// object_pose_tracking/{observation_node,cloud_node,pnp_gicp_node,pose_filter}.py.
// These stages describe the implementation; they do not simulate performance.

export const streams = [
  {
    id: 'dataset',
    title: 'Dataset collection + training',
    subtitle: 'Calibrated views become two models and one metric reference.',
    stages: [
      {
        title: 'Calibrate the object', short: 'Set the box', group: 'capture',
        detail: 'Temporary AprilTags define the object box. Permanent table tags preserve its box_center frame across camera views.',
        input: 'Object, setup tags and permanent table tags',
        output: 'Calibration, metric box dimensions and fixed object frame',
      },
      {
        title: 'Capture RGB-D views', short: 'Collect views', group: 'capture',
        detail: 'Remove setup tags and collect synchronized color and aligned depth. Each saved view includes intrinsics, its calibrated object pose and projected box corners.',
        input: 'Calibrated object and RGB-D camera',
        output: 'Color, aligned depth, intrinsics and T_camera_object per view',
      },
      {
        title: 'Review foreground masks', short: 'Review masks', group: 'dataset',
        detail: 'BiRefNet estimates foreground inside a pose-guided crop. Refinement removes unrelated foreground and tag regions; inspect every preview before export.',
        input: 'Captured color images and calibrated pose labels',
        output: 'Reviewed object masks paired with captured views',
      },
      {
        title: 'Export paired labels', short: 'Pair the labels', group: 'dataset',
        detail: 'Export matching segmentation polygons and eight ordered virtual box corners from the same views. Training replaces backgrounds on the fly and disables horizontal flips to preserve corner identity.',
        input: 'Reviewed masks, source images and projected box corners',
        output: 'Paired YOLO segmentation and eight-keypoint datasets',
      },
      {
        title: 'Fuse the metric reference', short: 'Build reference', group: 'reference',
        detail: 'A sibling branch fuses masked RGB-D captures with their saved poses into a TSDF reference directly in box_center. The metric reference is cropped to the calibrated box plus a small margin.',
        input: 'Reviewed masks, captured depth and calibrated view poses',
        output: 'reference.ply and frame metadata in box_center',
      },
      {
        title: 'Freeze geometry + train', short: 'Train two models', group: 'training',
        detail: 'Save a hash-checked snapshot of calibration and canonical 3D corners before training separate segmentation and pose models. The two checkpoints, frozen corners and sibling reference form the runtime bundle.',
        input: 'Paired datasets, fixed geometry and training backgrounds',
        output: 'Segmentation checkpoint, pose checkpoint and frozen PnP geometry',
      },
    ],
  },
  {
    id: 'pose',
    title: 'Pose estimation',
    subtitle: 'Use learned initialization, measured depth and accepted pose history.',
    stages: [
      {
        title: 'Acquire aligned RGB-D', short: 'Read the camera', group: 'capture',
        detail: 'The Gemini 305 supplies color and aligned depth with current rectified camera intrinsics. Synchronized observations share the color optical frame.',
        input: 'Live scene and camera calibration',
        output: 'Color, aligned depth and CameraInfo',
      },
      {
        title: 'Find the object', short: 'Infer mask + corners', group: 'inference',
        detail: 'YOLO segmentation selects and associates an object instance each frame. The separate eight-corner pose model runs for startup or recovery; established tracking skips pose inference.',
        input: 'Color frame and the two trained checkpoints',
        output: 'Selected foreground mask and corners when initialization is needed',
      },
      {
        title: 'Extract observed geometry', short: 'Build object cloud', group: 'cloud',
        detail: 'Backproject masked aligned depth using current intrinsics. Depth limits, voxel sampling and outlier removal produce an observed object cloud in the camera optical frame.',
        input: 'Synchronized mask, aligned depth and camera intrinsics',
        output: 'Filtered observed object cloud in metres',
      },
      {
        title: 'Choose the pose seed', short: 'Initialize or track', group: 'pnp',
        detail: 'Startup and recovery use robust PnP from detected corners and frozen 3D corners, with depth validation and bounded seed correction. During tracking, the previous accepted raw GICP pose supplies the seed.',
        input: 'Ordered corners + frozen geometry, or accepted raw pose history',
        output: 'Initial T_camera_object for local registration',
      },
      {
        title: 'Refine against the reference', short: 'Refine with GICP', group: 'refine',
        detail: 'GICP aligns the fixed metric reference to the observed cloud from the chosen seed. The current configuration enables the fine level; acceptance checks geometry quality, initialization agreement and temporal jumps.',
        input: 'Pose seed, fixed box_center reference and observed cloud',
        output: 'A refined pose candidate with quality checks',
      },
      {
        title: 'Accept, smooth + publish', short: 'Publish accepted pose', group: 'output',
        detail: 'Only accepted candidates update the raw tracking seed and temporal filter. Publish filtered pose, TF and projected box; rejected frames omit the old box. Repeated failures trigger PnP recovery.',
        input: 'Validated GICP candidate and tracking state',
        output: 'Filtered object pose, box_center TF and camera box overlay',
      },
    ],
  },
];

export const artifacts = [
  { name: 'Segmentation checkpoint', detail: 'training/yolo/SESSION_onthefly_seg/weights/best.pt selects the foreground object.' },
  { name: 'Eight-corner pose checkpoint', detail: 'training/yolo/SESSION_pose_refined_pose/weights/best.pt provides ordered image corners for startup and recovery.' },
  { name: 'Metric reference', detail: 'datasets/SESSION/references/CATEGORY/reference.ply is fused in box_center. The launcher prefers reference_aligned.ply when present.' },
  { name: 'Frozen PnP geometry', detail: 'datasets/SESSION/training_pnp/ stores calibration.json, object_keypoints.yaml and a hash manifest; the launcher verifies this snapshot and the reference frame.' },
];

export const contracts = [
  'Reference geometry, canonical corners and output TF share the calibrated box_center frame.',
  'P_camera = T_camera_object × P_object; translations and reference coordinates use metres.',
  'The camera frame is camera_color_optical_frame; runtime uses the current rectified intrinsics and aligned depth.',
  'Eight immutable corner IDs follow XYZ signs: 0 −−−, 1 +−−, 2 −+−, 3 ++−, 4 −−+, 5 +−+, 6 −++, 7 +++.',
  'The TSDF reference is built alongside model training; it is a measured geometry artifact, not a trained model.',
  'Only accepted raw GICP poses seed tracking; only accepted candidates enter smoothing.',
];

export function stageState(streamIndex, position) {
  const stream = streams[streamIndex === 1 ? 1 : 0];
  const value = Number(position);
  const index = Math.max(0, Math.min(5, Math.round(Number.isFinite(value) ? value : 0)));
  return { index, stage: stream.stages[index], progress: index / 5 };
}
