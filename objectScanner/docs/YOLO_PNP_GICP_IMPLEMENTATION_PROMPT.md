# YOLO Pose/Segmentation → PnP → GICP 6D Object Pose Pipeline

Status: proposed implementation idea; not yet implemented.

This document preserves the implementation prompt supplied on 2026-09-24. It
supersedes neither the implemented Phase 1–3 tracker nor its documentation until
the new architecture is explicitly implemented and validated. The supplied text
ends partway through the implementation-order list at item 2; that truncation is
preserved rather than completed speculatively.

## Task: Implement YOLO Pose/Segmentation → PnP → GICP 6D Object Pose Pipeline

Inspect the existing `objectScanner` repository first.

Do not rewrite working components unnecessarily. Reuse the existing ROS 2, RGB-D,
YOLO segmentation, point-cloud, TF2, ICP/GICP, and visualization infrastructure
wherever possible.

The new target architecture is:

```text
RGB
 │
 ▼
YOLO Pose / Segmentation
 │
 ├──────────── Mask ────────────────┐
 │                                  │
 └──────── 2D Keypoints             │
              │                     │
              ▼                     │
             PnP                    │
              │                     │
        Initial 6D Pose             │
        T_camera_object             │
              │                     │
              │              Depth + Mask
              │                     │
              │                     ▼
              │              Observed Object
              │               Point Cloud
              │                     │
              └──────────┬──────────┘
                         ▼
                        GICP
                         │
                         ▼
                 Refined 6D Pose
                         │
                  Quality Validation
                         │
                         ▼
                   Pose Filtering
                         │
                         ▼
                  T_camera_object
                         │
                        TF2
                         │
                         ▼
                   T_base_object
```

The object does NOT have a CAD model.

We have a high-quality complete 360-degree reference point cloud created by
scanning the real physical object from multiple viewpoints.

The final system should target approximately 10 Hz where hardware permits.

---

## 1. Design Principle

Separate the responsibilities clearly:

### YOLO

YOLO handles:

```text
object detection
segmentation
2D object keypoint detection
```

It should NOT be responsible for final metric 6D pose accuracy.

### PnP

PnP converts:

```text
known object-frame 3D keypoints
+
detected image-space 2D keypoints
+
camera intrinsics
```

into:

```text
T_camera_object_initial
```

PnP is the geometric 6D pose initializer.

### GICP

GICP uses:

```text
T_camera_object_initial
+
reference point cloud
+
observed RGB-D point cloud
```

to produce:

```text
T_camera_object_refined
```

GICP is responsible for final geometric refinement.

---

## 2. Canonical Object Coordinate Frame

This is one of the most important parts of the implementation.

Define one canonical:

```text
object_frame
```

for the scanned reference point cloud.

All components must use exactly this coordinate system:

```text
reference point cloud
3D keypoints
YOLO training labels
PnP
GICP
TF output
robot target
```

The transform convention must be:

```text
P_camera =
    T_camera_object *
    P_object
```

Therefore:

```text
T_camera_object
```

maps coordinates from `object_frame` into `camera_frame`.

Document this convention prominently in the code.

---

## 3. Define 3D Object Keypoints

Create a configuration describing known 3D keypoints in `object_frame`.

Do NOT hardcode keypoints into algorithm code.

For example:

```yaml
object_keypoints:

  - id: 0
    name: corner_0
    xyz: [x0, y0, z0]

  - id: 1
    name: corner_1
    xyz: [x1, y1, z1]

  ...
```

Initially support approximately 8–16 keypoints.

The implementation should allow arbitrary keypoint count.

Possible keypoints include:

```text
3D bounding-box corners
distinct geometric landmarks
manually selected reference-cloud landmarks
```

Keep keypoint IDs/order deterministic because the correspondence:

```text
3D keypoint i ↔ YOLO 2D keypoint i
```

must never become ambiguous.

---

## 4. Keypoint Visualization Tool

Create an offline utility for selecting/inspecting 3D keypoints on the reference
point cloud if the repository does not already contain one.

Desired workflow:

```text
reference.pcd
     ↓
3D viewer
     ↓
select points
     ↓
save
     ↓
object_keypoints.yaml
```

At minimum provide a simple method to load, visualize, and validate configured
keypoints relative to the reference cloud.

---

## 5. YOLO Pose + Segmentation Interface

Inspect the currently used YOLO implementation/version before changing dependencies.

Determine the cleanest supported way to obtain both:

```text
segmentation mask
2D keypoints
```

If one network can provide both reliably, prefer that.

If the existing framework requires separate heads/models, design the interface so
the rest of the pipeline does not care.

Expose a unified result conceptually similar to:

```cpp
struct ObjectObservation
{
    Mask mask;

    std::vector<Keypoint2D> keypoints;

    float detection_confidence;

    BoundingBox bbox;
};
```

Each keypoint should contain at least:

```text
u
v
confidence
valid
```

Do not send low-confidence keypoints blindly into PnP.

---

## 6. PnP Pose Estimation

Use the detected 2D keypoints and known 3D object keypoints.

Inputs:

```text
3D object keypoints
2D image keypoints
camera intrinsic matrix K
distortion coefficients
```

Estimate:

```text
R_camera_object
t_camera_object
```

and construct:

```text
T_camera_object_initial
```

Prefer robust PnP.

Evaluate appropriate OpenCV methods such as:

```text
solvePnPRansac
SOLVEPNP_ITERATIVE
SOLVEPNP_EPNP
```

Choose the method based on the keypoint geometry and document the reason.

Do not assume every configured keypoint is visible.

Only use valid keypoints above a configurable confidence threshold.

---

## 7. PnP RANSAC

Use RANSAC where appropriate to reject incorrect YOLO keypoints.

Track:

```text
number of valid keypoints
number of PnP inliers
reprojection error
```

Reject PnP if quality is insufficient.

Expose ROS parameters such as:

```text
keypoint_confidence_threshold
min_valid_keypoints
pnp_ransac_iterations
pnp_reprojection_threshold
pnp_confidence
min_pnp_inliers
max_mean_reprojection_error
```

---

## 8. Reprojection Validation

After PnP estimates:

```text
T_camera_object_initial
```

project the known 3D keypoints back into the image.

Calculate:

```text
observed keypoint
        ↕
reprojected keypoint
```

and compute reprojection errors.

Publish/debug:

```text
mean reprojection error
median reprojection error
maximum reprojection error
PnP inlier count
```

Create a debug image showing:

```text
YOLO keypoints
PnP reprojected keypoints
```

using different markers.

This will be important for debugging training quality.

---

## 9. Use Depth to Validate PnP Translation

Because this is an RGB-D system, do not rely only on monocular PnP.

Use the segmentation mask and depth image to estimate the approximate object
depth/3D centroid.

Compare it against:

```text
translation(T_camera_object_initial)
```

Reject or flag obviously inconsistent PnP estimates.

Implement this as a configurable validation step rather than forcing it into the
PnP solver itself.

---

## 10. Observed Point Cloud

Generate the runtime object cloud from:

```text
Depth
+
CameraInfo
+
YOLO segmentation mask
```

Pipeline:

```text
Depth
  ↓
Apply mask
  ↓
Depth → XYZ
  ↓
Remove invalid points
  ↓
Depth filtering
  ↓
Voxel downsample
  ↓
Optional outlier filtering
  ↓
Observed Object Point Cloud
```

Do not include background points outside the segmentation mask.

---

## 11. Reference Point Cloud

Load the complete scanned reference point cloud once during initialization.

The cloud is expressed in:

```text
object_frame
```

Precompute downsampled versions where useful.

Suggested configurable levels:

```text
coarse
medium
fine
```

Do not hardcode voxel sizes.

For example, initial values may be around:

```text
10 mm
5 mm
2 mm
```

but these must be parameters and tuned to the actual object size and sensor noise.

---

## 12. PnP → GICP Initialization

PnP gives:

```text
T_camera_object_initial
```

Use this to initialize point-cloud registration.

Conceptually:

```text
Reference Point Cloud
(object coordinates)
        │
        │ T_camera_object_initial
        ▼
Reference Cloud
(camera coordinates)
        │
        ▼
approximately aligned with
observed RGB-D cloud
        │
        ▼
GICP refinement
```

Avoid global registration such as FPFH + RANSAC during normal operation.

The neural/keypoint pipeline is now responsible for global initialization.

---

## 13. Coarse-to-Fine GICP

Implement GICP refinement using a coarse-to-fine strategy if practical:

```text
T_PnP
   ↓
Coarse GICP
   ↓
T_coarse
   ↓
Medium GICP
   ↓
T_medium
   ↓
Fine GICP
   ↓
T_refined
```

Each level should support configurable:

```text
voxel size
max correspondence distance
max iterations
transformation epsilon
fitness threshold
```

Measure whether all three levels are necessary for 10 Hz.

Allow levels to be individually disabled.

---

## 14. Partial-to-Full Registration

Remember:

```text
reference cloud = complete 360° object
observed cloud  = only camera-visible surfaces
```

Therefore this is a partial-to-full registration problem.

Do not assume both point clouds contain identical geometry.

Implement a modular:

```text
ReferenceVisibilityFilter
```

that receives:

```text
reference cloud
T_camera_object_initial
camera intrinsics
```

and returns an approximation of the currently visible reference surface.

Initial implementation may use:

```text
positive depth filtering
camera FOV filtering
surface-normal facing test
```

Design it so z-buffer / occlusion filtering can be added later.

Allow visibility filtering to be disabled for benchmarking.

---

## 15. Runtime Tracking

Do not necessarily run PnP as the only initializer every frame.

Once GICP has successfully established a high-quality pose:

```text
T_camera_object(t-1)
```

use it as the primary initial guess for frame `t`.

```text
Frame t-1

T_refined(t-1)
       │
       ▼
initial guess
       │
       ▼
GICP(frame t)
       │
       ▼
T_refined(t)
```

This should normally be faster and smoother.

---

## 16. YOLO/PnP as Reinitialization

Use YOLO keypoints + PnP when:

```text
startup
tracking is lost
GICP quality fails
pose jump is detected
object disappears and reappears
```

Suggested state machine:

```text
             ┌────────────────┐
             │ INITIALIZATION │
             └───────┬────────┘
                     │
              YOLO + PnP
                     │
                   GICP
                     │
                  valid?
                     │
                     ▼
               ┌──────────┐
               │ TRACKING │
               └────┬─────┘
                    │
          previous pose → GICP
                    │
               quality bad
                    │
                    ▼
                ┌──────┐
                │ LOST │
                └───┬──┘
                    │
                YOLO + PnP
                    │
                    └────→ INITIALIZATION
```

However, keep YOLO segmentation available whenever it is needed to generate the
object mask.

Separate:

```text
YOLO segmentation inference requirement
```

from:

```text
PnP initialization requirement
```

in the architecture.

---

## 17. Pose Quality Validation

Validate both stages independently.

### PnP quality

Evaluate:

```text
keypoint confidence
number of valid keypoints
PnP inliers
reprojection error
depth consistency
```

### GICP quality

Evaluate:

```text
fitness
RMSE
overlap
correspondence count
```

### Temporal quality

Calculate:

```text
DeltaT =
    inverse(T_previous) *
    T_current
```

and extract:

```text
translation_delta
rotation_delta
```

Reject implausible pose jumps.

All thresholds must be configurable.

---

## 18. Pose Filtering

After validation, optionally filter the final pose.

Initial implementation:

```text
translation:
EMA

rotation:
quaternion SLERP
```

Keep this modular so it can later be replaced by:

```text
One Euro Filter
Kalman Filter
EKF
```

Never filter invalid registration results into valid tracking data.

---

## 19. TF Structure

Maintain clear TF conventions.

Desired structure:

```text
base_link
    │
    ▼
camera_frame
    │
    ▼
object_frame
```

The perception pipeline estimates:

```text
T_camera_object
```

TF2 provides:

```text
T_base_camera
```

Therefore:

```text
T_base_object =
    T_base_camera *
    T_camera_object
```

Publish/broadcast the refined object transform.

---

## 20. Debug TF Frames

During development optionally expose:

```text
object_pnp
object_gicp
object_filtered
```

so RViz can visually compare:

```text
PnP estimate
vs
GICP refinement
vs
filtered final pose
```

This should be switchable off in production.

---

## 21. ROS 2 Outputs

Inspect existing topics first and reuse them where appropriate.

Conceptually provide:

```text
/object_pose/pnp
/object_pose/gicp
/object_pose/filtered
/object_pose/base
```

Also expose status information such as:

```text
tracking_state
pose_valid
PnP reprojection error
PnP inlier count
GICP fitness
GICP RMSE
GICP overlap
```

Do not create redundant topics if the repository already has suitable interfaces.

---

## 22. RViz Debugging

Provide an RViz-friendly debugging setup showing:

```text
camera image
segmentation mask
detected keypoints
PnP reprojections

observed object point cloud
reference point cloud
PnP-transformed reference
GICP-aligned reference

camera_frame
object_pnp
object_gicp
object_filtered
base_link
```

This is important because we need to visually identify whether failures originate
from:

```text
YOLO
PnP
depth
coordinate conventions
GICP
TF
```

---

## 23. Performance Instrumentation

Target:

```text
~10 Hz
```

Measure execution time separately for:

```text
YOLO inference
point-cloud generation
PnP
visibility filtering
GICP coarse
GICP medium
GICP fine
pose validation
total pipeline
```

Report moving-average execution times.

Avoid unnecessary:

```text
CPU/GPU transfers
point-cloud copies
reference-cloud processing
memory allocations
```

Precompute all static reference information.

---

## 24. Dataset / Training Preparation

Do not assume the existing YOLO dataset already contains the required pose keypoints.

Inspect the dataset tooling.

Design the training-data convention so every sample contains consistent:

```text
segmentation annotation
2D keypoints
keypoint visibility
object identity
```

The keypoint order must correspond exactly to:

```text
object_keypoints.yaml
```

---

## 25. Automatic Keypoint Label Generation

Design for future automatic label generation.

If camera pose relative to the object is known during dataset capture, generate 2D
labels from the known 3D keypoints:

```text
P_object
    │
    │ ground-truth T_camera_object
    ▼
P_camera
    │
    │ camera intrinsics K
    ▼
[u,v]
```

Projection:

```text
P_camera =
    T_camera_object *
    P_object

u = fx * X/Z + cx
v = fy * Y/Z + cy
```

Use this to avoid manually annotating every 2D keypoint where ground-truth
camera/object transforms are available.

Handle visibility/occlusion correctly rather than labeling invisible points as
visible.

---

## 26. Object Symmetry

Add explicit support/documentation for object symmetry.

Some objects may have poses that are geometrically equivalent under:

```text
90°
180°
continuous axial rotation
```

Do not assume every object has a unique orientation.

Create a configuration concept such as:

```yaml
symmetry:
    type: none
```

with future support for:

```text
discrete rotational symmetry
continuous axial symmetry
```

At minimum document where symmetry-aware pose validation/loss handling would need
to occur.

---

## 27. Offline Registration Test

Create a tool that can run without the live ROS pipeline.

Inputs:

```text
reference.pcd
observed.pcd
T_initial
```

Output:

```text
T_refined
fitness
RMSE
overlap
runtime
```

Allow visualization of:

```text
before GICP
after GICP
```

This will be used to tune GICP independently from YOLO/PnP.

---

## 28. Offline PnP Test

Also create a PnP test utility.

Inputs:

```text
object_keypoints.yaml
2D detected/ground-truth keypoints
camera intrinsics
```

Outputs:

```text
T_camera_object
PnP inliers
reprojection error
```

Visualize the reprojected keypoints.

This will allow debugging PnP independently from GICP.

---

## 29. Important Failure Modes

Add logging/diagnostics for:

```text
insufficient visible keypoints
bad keypoint confidence
wrong keypoint ordering
PnP degeneracy
high reprojection error
PnP/depth inconsistency
incorrect object coordinate frame
insufficient depth points
segmentation contamination
GICP local minimum
low cloud overlap
symmetrical pose ambiguity
tracking jump
TF timestamp mismatch
```

Failures should be identifiable from logs rather than appearing simply as
"registration failed".

---

## 30. Implementation Order

Before coding:

1. Inspect repository architecture.
2.
