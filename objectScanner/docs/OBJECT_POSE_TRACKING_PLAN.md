# ROS2 YOLO Segmentation + RGB-D Fixed-Reference Object Pose Tracking Plan

Status (updated 2026-09-23): Phases 1–3 implemented. Phase 2 includes independent
centroid-seeded fixed-reference ICP, pose/TF, expiring XYZ frame markers, and the
aligned reference overlay. Synthetic registration and ROS output tests pass;
physical frame/overlay acceptance remains pending. Phase 3 adds last-accepted raw
pose seeding with seed expiration and verified synthetic small-motion sequences.
Phase 4 and later remain unimplemented. See [Phase 1](PHASE1_RGBD_SEGMENTATION.md),
[Phase 2](PHASE2_FIXED_REFERENCE_ICP.md), and
[Phase 3 implementation and validation](PHASE3_PREVIOUS_POSE_TRACKING.md).
Created: 2026-09-23.
Project: `/home/jb/codex/projects/objectScanner`.

This document preserves the complete implementation prompt from the conversation
“เปรียบเทียบ SVD กับ ICP”, followed by repository-specific integration notes.
The source prompt and original planning-only instructions below are retained as
historical context; Phases 1–3 were subsequently authorized in implementation sessions.
The implementation instructions below describe future work; creating this plan
does not authorize executing any implementation phase.

## Repository inspection and integration context

- The existing project is a standalone Python RealSense L515 scanner with
  Open3D reconstruction, plus a Flask/Three.js reference editor. No ROS2 package
  manifest, ROS2 launch/config structure, or existing `docs/` directory was found.
  No applicable `AGENTS.md` was found in the inspected ancestor directories.
  The directory is not currently recognized as a Git checkout.
- `README.md` is the existing English Markdown documentation. Python modules
  live at the project root and tests use `unittest` under `tests/`.
- `requirements.txt` pins NumPy 1.26.4, OpenCV 4.10.0.84, Open3D 0.19.0,
  pyrealsense2 2.54.2.5684, and Flask 3.1.3. The README documents a Python 3.10
  virtual environment. `/opt/ros/jazzy` exists, but ROS2 runtime usability and
  Python/dependency compatibility have not been verified. Select and verify the
  ROS2 environment before scaffolding; do not assume the existing venv is compatible.
- `camera.py` already aligns depth to rectified RGB and handles L515 stream
  selection and sensor depth scale. Its processed `RGBD.depth` is uint16
  millimetres, while exported point clouds and geometry use metres.
- `webui.py` and `reference_geometry.py` already support fixed-frame reference
  creation. Exports under `references/<name>/` contain `reference.ply`,
  `frame.json`, usage notes, and a bundle. The PLY is already in the chosen
  `obj_frame`, in metres. Treat that selected frame as the planned ROS `object`
  frame (or an explicitly configured equivalent); do not apply the export's
  scan-to-object transform a second time.
- Existing transforms use homogeneous column vectors:
  `p_scan = T_scan_obj @ p_obj`. Preserve this convention in the tracker.
- Keep scanning/reconstruction and reference editing usable. Reuse their geometry
  and camera contracts where appropriate, without repurposing the scanner's
  changing reconstruction model as the tracking reference.
- The README documents prior GPU freezes and native-call stalls, and uses CPU
  defaults. Preserve those existing defaults; benchmark any new YOLO/ICP device
  configuration explicitly rather than assuming GPU operation is validated.

## Complete source plan

I want you to implement a ROS2 object pose estimation and tracking system using RGB-D data, YOLO segmentation, and ICP point-cloud registration.

The final goal is to estimate the 6D pose of a known object relative to the camera and eventually transform that pose into the robot base frame for robotic manipulation.

## System Goal

The pipeline should be:

RGB-D Camera
→ YOLO Instance Segmentation
→ Segmentation Mask
→ Extract Object Point Cloud from Depth
→ Point Cloud Preprocessing
→ ICP Registration against a Fixed Reference Point Cloud
→ Pose Validation
→ Pose Filtering
→ Publish Object Pose and TF

Target runtime is approximately 10 Hz.

Use a modular ROS2 architecture so individual components can be tested independently.

---

## Coordinate Frames

Use the following frame convention:

- `camera_link`: camera coordinate frame
- `object`: fixed coordinate frame defined in the reference object point cloud
- `robot_base`: robot base frame
- `grasp_pose`: predefined grasp frame attached to the object

The reference point cloud MUST remain defined in the fixed `object` coordinate frame.

Do NOT transform or redefine the reference point cloud coordinate system every frame.

ICP should estimate:

    T_camera_object

which represents the pose of the fixed object frame relative to the camera frame.

Eventually the robot target pose will be:

    T_robot_grasp =
        T_robot_camera
        * T_camera_object
        * T_object_grasp

For the first implementation, focus on estimating and publishing:

    T_camera_object

Do not implement robot motion yet.

---

## Important ICP Tracking Strategy

This is NOT frame-to-frame ICP.

Do NOT register:

    observed_cloud[t-1] → observed_cloud[t]

Instead, every frame must register the SAME fixed reference model against the current observed point cloud:

    reference_object_cloud
        → current_observed_object_cloud

For the first successful detection, generate an initial transformation estimate.

At minimum, initialize translation using the difference between the reference-cloud centroid and observed-cloud centroid.

If necessary, design the code so a better coarse registration method can be added later.

After a successful ICP result at frame `t-1`, store:

    last_object_pose = T_camera_object[t-1]

For the next frame, use:

    initial_guess = last_object_pose

Then perform ICP:

    T_camera_object[t] =
        ICP(
            fixed_reference_cloud,
            current_observed_cloud,
            initial_guess
        )

Conceptually:

    Fixed Reference Cloud
            |
            | transform using T[t-1]
            v
    Initial Alignment
            |
            v
           ICP
            |
            v
          T[t]

This prevents us from relying on accumulated frame-to-frame registration and reduces drift.

---

## ROS2 Nodes

Please create the system using modular nodes.

### 1. YOLO Segmentation Node

Inputs:

    /camera/color/image_raw

Outputs:

    /object/mask

Responsibilities:

- Run YOLO instance segmentation.
- Detect the configured target object class.
- Generate a binary segmentation mask.
- Publish the mask with timestamp information compatible with the corresponding depth frame.
- Make model path, target class, confidence threshold, and device configurable through ROS2 parameters.

Keep the YOLO implementation isolated so the segmentation backend can be replaced later.

---

### 2. Object Cloud Extraction Node

Inputs:

    /object/mask
    /camera/depth/image_raw
    /camera/camera_info

Output:

    /object/cloud

Responsibilities:

- Synchronize mask and depth data appropriately.
- Use the segmentation mask to select only object depth pixels.
- Back-project depth pixels into 3D using camera intrinsics:

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

- Publish the resulting cloud in `camera_link`.

Handle invalid depth values correctly.

Add configurable preprocessing:

- minimum depth
- maximum depth
- voxel downsampling
- statistical outlier removal

Structure the implementation so optional plane removal can be added easily.

---

### 3. ICP Pose Estimation Node

Input:

    /object/cloud

Reference:

    Load a fixed reference point cloud from a `.pcd` or `.ply` file.

Outputs:

    /object/pose
    TF: camera_link → object

Responsibilities:

1. Load the reference cloud once during node initialization.
2. Downsample/preprocess the reference cloud once, not every frame.
3. Keep the reference cloud permanently expressed in the `object` frame.
4. Receive the current observed object cloud.
5. Generate an initial pose for the first valid frame.
6. Run ICP against the fixed reference model.
7. After successful registration, save the result as `last_object_pose`.
8. Use `last_object_pose` as the initial guess for the next frame.
9. Publish the resulting pose and TF.

Prefer Point-to-Plane ICP if reliable normals are available.

However, make the registration implementation modular so Point-to-Point ICP can also be selected through a ROS2 parameter.

---

## ICP Validation

Do NOT blindly accept every ICP result.

Track at least:

- convergence status
- fitness score
- RMSE
- number or ratio of valid correspondences
- ICP execution time
- pose change from the previous frame

Calculate the incremental motion:

    Delta_T =
        inverse(T[t-1]) * T[t]

Extract:

- translation magnitude
- rotation magnitude

Reject suspicious results based on configurable thresholds.

Example logic:

    if converged
       and fitness > fitness_threshold
       and rmse < rmse_threshold
       and correspondence_ratio > minimum_ratio
       and translation_delta < maximum_translation_jump
       and rotation_delta < maximum_rotation_jump:

        accept pose

    else:
        reject pose

Do not hard-code arbitrary threshold values throughout the source code. Expose them as ROS2 parameters.

If tracking repeatedly fails, reset tracking so the system returns to initial-pose estimation mode.

---

## Tracking State Machine

Implement a simple state machine:

    UNINITIALIZED
         |
         | valid segmented cloud
         v
    INITIALIZING
         |
         | successful ICP
         v
      TRACKING
         |
         | ICP success
         +--------------------+
         |                    |
         +---- next frame <---+
         |
         | repeated failures
         v
       LOST
         |
         | reinitialize
         v
    INITIALIZING

Do not continue publishing obviously invalid poses while tracking is LOST.

---

## Pose Filtering

Add a simple optional pose filter after ICP.

For the first implementation, a low-pass filter is sufficient.

Filter translation separately.

For orientation, do NOT average Euler angles directly. Use quaternion interpolation such as SLERP.

Keep the filtering implementation isolated so it can later be replaced with a Kalman Filter or EKF.

Pipeline:

    ICP Pose
       ↓
    Validation
       ↓
    Pose Filter
       ↓
    Published Pose / TF

---

## Performance Requirement

The target update rate is approximately:

    10 Hz

Therefore:

    total processing time < approximately 100 ms/frame

Optimize for realtime operation.

Important performance rules:

- Do not recreate the reference cloud every frame.
- Do not recompute static reference preprocessing every frame.
- Use voxel-downsampled clouds for ICP.
- Avoid unnecessary point-cloud copies.
- Use the previous valid pose as the ICP initial guess.
- Measure execution time for each important stage.

Publish or log timing information for:

    segmentation_time
    cloud_extraction_time
    preprocessing_time
    icp_time
    total_tracking_time

Do not sacrifice code correctness merely to reach 10 Hz. Make the timing measurable first so bottlenecks can be identified.

---

## Debugging / Visualization

Make the system easy to debug in RViz.

Publish:

    /object/cloud
    /object/reference_cloud_transformed
    /object/pose

and TF:

    camera_link
        ↓
      object

`/object/reference_cloud_transformed` should contain the fixed reference cloud transformed using the latest estimated `T_camera_object`.

When ICP works correctly, the transformed reference cloud should visually overlap the observed object cloud in RViz.

This is an important debugging feature.

---

## ROS2 Parameters

At minimum expose:

    reference_cloud_path
    camera_frame
    object_frame

    voxel_size

    icp_type
    icp_max_iterations
    icp_max_correspondence_distance
    icp_transformation_epsilon

    fitness_threshold
    rmse_threshold
    minimum_correspondence_ratio

    maximum_translation_jump
    maximum_rotation_jump

    lost_tracking_failure_count

    pose_filter_enabled
    pose_filter_alpha

    target_object_class
    yolo_model_path
    yolo_confidence_threshold
    yolo_device

Do not scatter configuration constants through the implementation.

---

## Code Architecture

Keep the implementation clean and modular.

Separate:

    segmentation
    cloud extraction
    cloud preprocessing
    registration
    pose validation
    pose filtering
    TF publishing

Avoid putting the entire pipeline inside one large ROS2 node or source file.

Use clear classes/interfaces so individual algorithms can later be replaced.

Add comments explaining transformation conventions.

Be especially explicit about whether a transform means:

    T_A_B = pose of frame B expressed in frame A

Use this convention consistently throughout the project.

---

## Implementation Strategy

Before changing code:

1. Inspect the existing repository.
2. Identify the ROS2 distribution, package structure, language, existing camera interfaces, dependencies, and coding conventions.
3. Reuse existing infrastructure where appropriate.
4. Do not unnecessarily rewrite working code.
5. Present a short implementation plan based on the actual repository.
6. Then implement incrementally.

Prioritize this MVP order:

Phase 1:
    RGB-D + segmentation
    → object point cloud
    → visualize `/object/cloud` in RViz

Phase 2:
    Load fixed reference cloud
    → initial centroid alignment
    → ICP
    → publish `camera_link → object`

Phase 3:
    Previous valid pose as ICP initial guess
    → realtime tracking

Phase 4:
    ICP validation
    → LOST/reinitialization behavior

Phase 5:
    pose filtering
    → timing/performance optimization toward 10 Hz

Phase 6:
    robot-base transform and grasp frame integration

Do NOT implement robot movement in the initial version.

---

## Tests

Add tests where practical, especially for transformation math.

Create synthetic tests where a reference cloud is transformed using a known transformation:

    T_ground_truth

Then run registration and verify:

    T_estimated ≈ T_ground_truth

Measure separately:

    translation_error
    rotation_error

Also test:

- identity transformation
- translation only
- rotation only
- combined rotation + translation
- noisy cloud
- partial observation
- tracking with small frame-to-frame motion
- ICP failure / invalid cloud
- recovery after tracking loss

---

## Final Expected Behavior

When the system starts:

    Camera
       ↓
    YOLO segmentation
       ↓
    Current object cloud
       ↓
    Initial pose estimation
       ↓
    ICP against FIXED reference
       ↓
    T_camera_object

For subsequent frames:

    Current object cloud[t]
            ↑
            |
    ICP(initial_guess = T[t-1])
            ↑
            |
    SAME fixed reference cloud

producing:

    T[0], T[1], T[2], ...

The fixed object reference frame must never drift.

The previous frame pose is ONLY the initial guess for the next registration.

Once implemented, provide:

1. A summary of files created/modified.
2. ROS2 build commands.
3. Required dependencies.
4. Example launch command.
5. Example parameter YAML.
6. RViz topics and TF frames to inspect.
7. Instructions for supplying the reference `.pcd/.ply`.
8. Current limitations.
9. Measured or expected performance bottlenecks.
10. Recommended next steps for reaching stable 10 Hz tracking.

## Repository-specific implementation clarifications

These details make the source plan concrete without changing its fixed-reference
registration or coordinate-frame requirements.

### Input and transform contracts

- Choose a compatible ROS2 camera driver or a small adapter around the existing
  acquisition interface during implementation. Preserve RGB/depth alignment,
  matching intrinsics after resize, and acquisition timestamps through inference.
  The listed camera topic names are defaults to remap to the chosen source.
- Back-projection produces optical coordinates (X right, Y down, Z forward).
  Verify the actual camera driver's frames. If these differ from `camera_link`,
  transform points with the calibrated optical-to-`camera_link` TF at the
  measurement timestamp before publishing `/object/cloud`; never merely relabel
  optical coordinates as `camera_link`.
- Explicitly convert depth into metres before back-projection. When using
  `camera.py` processed depth, multiply millimetres by 0.001; for another source,
  use its documented encoding and scale. Reject zero, non-finite, and out-of-range
  values. Synchronization tolerance, bounded queue sizes, depth limits, outlier
  removal, and minimum point count must be configurable.
- Use `sensor_msgs/Image` for masks, `sensor_msgs/PointCloud2` for clouds, and
  `geometry_msgs/PoseStamped` for poses. Pose and TF must describe the same
  accepted output, parent frame, child frame, and measurement timestamp.
- Select a single target instance consistently when several detections share a
  class. Make selection behavior explicit so the tracker does not silently jump
  between different objects.
- Reference data remains immutable after startup preprocessing. Only temporary
  debug copies are transformed into camera coordinates.
- For first initialization with identity rotation, set
  `t0 = centroid(observed) - centroid(reference)`. If a rotation guess is supplied,
  use `t0 = centroid(observed) - R0 @ centroid(reference)`.
  Centroid alignment alone does not resolve unknown orientation, symmetry, or
  strong partial visibility; keep coarse initialization replaceable.
- Define the registration interface to return the absolute source-to-target
  `T_camera_object`. If the backend already returns that absolute transform,
  do not multiply it by the initial guess again. If a backend reports a correction
  to a pretransformed source, compose it explicitly within the adapter.
- Store the last validated raw ICP result for the next initial guess. Filtering
  only changes the published output; it does not mutate the reference or turn
  tracking into registration between successive observed clouds.

### Validation, loss, and output behavior

- Validate finite rigid transforms, sufficient valid points, correspondence
  quality, and convergence evidence before accepting any pose. Document how the
  selected backend defines fitness, RMSE, correspondence ratio, and termination;
  adapt backend-specific values to the plan's validation contract.
- Apply motion-jump gates only when a previous valid pose exists. Rejected
  candidates must not replace it. Increment failure count on unusable frames or
  registration failure; reset the count on successful validation.
- Missing detections or stalled input also require a configurable data-age timeout,
  so tracking cannot remain valid indefinitely when no callbacks arrive.
- On the configured failure count or timeout, enter `LOST`, suppress new
  pose/TF output, and invalidate the prior initialization/filter state. Reinitialize
  on a new usable segmented cloud. Publish tracking state and failure reasons
  through diagnostics or an explicit status interface.
- Never refresh stale poses with new timestamps. Previously published TF may still
  exist in consumers' caches, so downstream users must check state and age.
- For optional filtering, define
  `p_out = (1 - alpha) * p_out_previous + alpha * p_accepted` and
  `q_out = SLERP(q_out_previous, q_accepted, alpha)`, with normalized quaternions,
  shortest-arc sign handling, and `0 < alpha <= 1`.
  Initialize/reset the filter from the first newly accepted pose after acquisition.
- Publish the reference debug overlay using the same published pose/TF for visual
  consistency, at a configurable rate. Label any additional raw-pose overlay
  separately if introduced.

### Phase checkpoints and completion criteria

All phases are pending. Preserve the six-phase order above; basic input/rigid-pose
checks start with registration, while full quality gates and recovery belong to
Phase 4.

| Phase | Acceptance checkpoint |
| --- | --- |
| 1 — RGB-D and segmentation | Synchronized target mask produces a valid cloud in metres and the correct camera frame; inspect `/object/cloud` in RViz. |
| 2 — Fixed reference and first pose | Load/preprocess the reference once; centroid-seeded ICP produces a source-to-target pose, correct TF, and an overlapping reference debug cloud; synthetic transform direction tests pass. |
| 3 — Realtime tracking | Every update uses the unchanged reference and current observation, seeded only by the last validated pose; record timing and demonstrate small-motion sequences. |
| 4 — Validation and recovery | Poor fits and jumps are rejected; failures and stale data enter LOST; new observations can reinitialize; no invalid or refreshed stale pose is emitted. |
| 5 — Filtering and performance | Optional translation low-pass and quaternion SLERP behave correctly; measure stage times, end-to-end latency, and achieved update rate while tuning toward approximately 10 Hz. |
| 6 — Robot and grasp frames | Verify `T_robot_camera * T_camera_object * T_object_grasp` and TF integration using known transforms; robot motion remains outside this plan's implementation scope. |

The original conversation recommends an initial implementation checkpoint after
Phase 2: verify the cloud/TF/overlay in RViz before proceeding to realtime tracking.
Do not begin any phase in the current documentation-only task.

For future implementation, use bounded queues and drop superseded observations
rather than building an ever-growing latency backlog. Measure acquisition-to-output
latency separately from stage execution time and throughput. Approximately 10 Hz
and approximately 100 ms/frame are targets, not measured results or hard realtime
guarantees.

Synthetic tests should use a nondegenerate asymmetric reference and controlled
initial guesses, report translation error in metres and rotation error in radians,
and cover every case in the source plan. Also check reference immutability,
transform composition/direction, rejection leaving the last valid pose unchanged,
filter reset and quaternion sign handling, and stale-input recovery. Hardware
integration and RViz checks are separate from deterministic synthetic tests.

Before each future implementation session, read this file, re-inspect the project,
and implement only the next agreed unfinished phase. Explain any proposed change
to the coordinate-frame or fixed-reference ICP conventions before making it.
