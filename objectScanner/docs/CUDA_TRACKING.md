# CUDA object tracking

Phase 3 now supports CUDA YOLO inference and CUDA ICP using the supplied weights
and fixed reference. RGB-colored clouds, the aligned reference, estimated object
axes, pose/TF timestamps, and interactive RViz controls are unchanged.

The optimized GPU-resident point-to-plane implementation is now faster than
cached CPU ICP on the supplied reference in paired synthetic tests: 4.13 ms
versus 6.04 ms median solver time. The original CUDA loop remains selectable
as `cuda_legacy`.

The CPU environment and launchers remain available. CUDA has a separate Python
3.12 environment and colcon install so generated ROS executables use the correct
interpreter. The existing Python 3.10 camera worker is shared by both versions.
No driver or system CUDA changes are required by this setup.

## Install and run

From the project root:

```bash
python3 -m venv --system-site-packages .venv-ros-cuda
.venv-ros-cuda/bin/python -m pip install -r requirements-ros-cuda.txt
./scripts/build_cuda.sh
./scripts/run_phase3_cuda.sh
```

Stop the existing camera launch before starting another. Ctrl+C stops the launch
group. The default CUDA launcher puts **both YOLO and ICP on CUDA:0**. Add
`rviz:=false` for headless operation. RViz retains software rendering, independent
of CUDA compute. Reference/weights/parameter launch arguments work as in Phase 3.

CUDA YOLO plus cached CPU ICP remains available as a hybrid preset for
comparison or workloads where CPU ICP is preferable. It was faster than the
first CUDA implementation; the revised CUDA point-to-plane loop now wins the
controlled comparison for the supplied reference. Run the hybrid preset with:

```bash
./scripts/run_phase3_cuda.sh \
  device_params_file:="$PWD/ros2_ws/src/object_pose_tracking/config/devices_hybrid.yaml"
```

To return to the original CPU environment, stop the CUDA launch and run
`./scripts/run_phase3.sh`.

Device overrides load after the regular phase parameters. Copy
[`devices_cuda.yaml`](../ros2_ws/src/object_pose_tracking/config/devices_cuda.yaml)
to customize devices or precision, then pass its absolute path through
`device_params_file`. The default device override for ordinary launchers is empty
and therefore preserves their existing custom parameter choices.

| Parameter | CUDA preset | Meaning |
| --- | --- | --- |
| `yolo_device` | `cuda:0` | PyTorch device; `cpu` or a GPU index also accepted |
| `yolo_half` | `false` | Optional FP16 inference; requires CUDA |
| `icp_backend` | `cuda` | Optimized CUDA, `cuda_legacy`, `cached` CPU, or `open3d` CPU |
| `icp_cuda_device` | `0` | Open3D CUDA device index |

FP32 inference remains the default; FP16 can change masks and needs validation on
representative images. Missing CUDA support fails explicitly during startup.
There is no silent CPU fallback. Diagnostics report the inference device/precision
and ICP backend. Unexpected CUDA native runtime errors propagate out of the node
and stop the launch group; the known singular point-to-plane system is rejected
as a geometric failure. This does not make a driver-level GPU hang recoverable.
The repository's previous GPU bus fault remains relevant; the tests below are
short validations, not a long-duration stability result.

## Implementation

- YOLO uses CUDA through PyTorch/Ultralytics. Prediction explicitly synchronizes
  the CUDA device before returning, so timing includes GPU execution; masks and
  boxes return to CPU for the existing same-timestamp ROS mask publication.
- The default CUDA point-to-plane loop keeps the reference, target, correspondences,
  transform, normal equations and routine validation on GPU. PyTorch compiles fused
  Float64 reductions, and CUDA graphs replay them. This avoids expensive generic
  tiny matrix multiplications and repeated transfer of correspondence arrays.
- Exact Triton search scans target tiles for clouds of up to 8,192 points, without
  allocating a full source-by-target distance matrix. Radius comparison is strict;
  equal distances choose the lowest target index. Larger targets use an Open3D
  CUDA search index with DLPack sharing. Search buffers never pass through CPU.
- One 72-byte status block returns per iteration for quality/convergence decisions
  and the elapsed-time budget. The final 4x4 pose returns at completion. The
  observability check uses a conservative GPU positive-definiteness test on the
  centered Jacobian's Gram matrix. If it cannot prove the existing singular-value
  ratio threshold is satisfied, the original direct CPU SVD decides that case.
  This uncommon path downloads the Jacobian rather than weakening the rank gate.
- Compilation and graph capture happen once at engine initialization, before
  processing measurements. A cold compiler cache increases startup time; allow
  startup to finish. The target buffers and graphs are reused across observations.
  The resident workspace accommodates 131,072 target points (or the reference
  count if larger); bigger preprocessed targets use the compatibility CUDA loop.
- `cuda_legacy` retains the original Open3D tensor loop for direct comparisons.
  Point-to-point registration also retains that implementation; the GPU-resident
  optimization specifically targets the default point-to-plane method.
- Camera processing, cloud extraction/downsampling, normal estimation and ROS/RViz
  output remain on CPU. Their costs are included in end-to-end latency, while the
  ICP benchmarks below report the solver separately. All previous pose-seeding,
  fitness/RMSE, rigid/observability, iteration/time and stale-data gates remain.

API references: [Ultralytics prediction options](https://docs.ultralytics.com/modes/predict/),
[Open3D tensor transformation estimators](https://www.open3d.org/docs/0.19.0/python_api/open3d.t.pipelines.registration.TransformationEstimationPointToPlane.html),
[Open3D native search](https://www.open3d.org/docs/0.19.0/python_api/open3d.core.nns.NearestNeighborSearch.html).

## Validation on this machine (2026-09-23)

Environment: RTX 4070 Laptop GPU, NVIDIA driver 595.84, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, Open3D 0.19.0 with CUDA support, Ultralytics 8.3.253.
The exact package requirements are in `requirements-ros-cuda.txt`.

Both CPU and CUDA ROS packages build. The CUDA environment passes **92 tests**,
including real GPU point/plane pose recovery with partial observations,
reference immutability/reuse, no correspondences, planar degeneracy, quality and
execution limits, and ROS output clearing/propagation on native failure. The CPU
environment passes 77 tests with 15 hardware-specific tests skipped. Additional
coverage checks exact tiled search, padding, ties, strict radius boundaries,
small/large target switching, graph/reference reuse and conservative GPU rank
decisions against direct SVD.

```bash
./scripts/test_cuda.sh -q
./scripts/test_phase1.sh -q
```

`test_cuda.sh` isolates test publishers on ROS domain 84 and bounds the full test
process to 180 seconds. A genuine driver/kernel hang may resist process shutdown.

A paired YOLO comparison used one actual 960x540 camera RGB image, three warmups,
and 20 alternating CPU/GPU repetitions, with identical image size and FP32:

| Inference, including transfers | CPU | CUDA |
| --- | ---: | ---: |
| Median | 49.26 ms | 8.10 ms |
| p95 | 52.22 ms | 10.09 ms |

Neither device detected the object in that image, so this establishes timing on
that input, **not target-mask accuracy**. [Raw report](validation/cuda_segmentation_20260923.json).

The revised paired ICP comparisons each used 60 deterministic observations of
this reference, two repetitions, alternating execution order, warm and centroid
seeds, and full/partial/noisy surfaces. Both backends receive identical inputs.
The benchmark budget is 2 seconds for equal-work comparison; production remains
250 ms. Measurements exclude engine initialization/compilation; each fit still
includes its target upload and final pose transfer in solver timing.

| Current ICP comparison | Cached CPU | GPU-resident CUDA |
| --- | ---: | ---: |
| Median solver time | 6.04 ms | 4.13 ms |
| p95 | 57.60 ms | 35.68 ms |
| Accepted / attempted | 106 / 120 | 106 / 120 |

CUDA reduced median solver time by 32% and p95 by 38% in this run. Preparation
was about 3.0 ms for either backend. Acceptance, rejection reasons and iteration
counts matched, with maximum accepted pose differences below 3e-16 m and
2e-15 rad. [Raw report](validation/cuda_resident_vs_cpu_20260923.json) and
[profile](validation/cuda_resident_profile_20260923.txt).

A separate paired comparison against the first CUDA loop measured 9.80 → 4.15 ms
median (58% reduction) and 90.00 → 31.07 ms p95. Both again accepted 106/120 trials
and rejected the same 14 for iteration limits.
[Raw report](validation/cuda_resident_vs_legacy_20260923.json).
The earlier [CUDA report](validation/cuda_icp_20260923.json) describes the
compatibility loop and is retained as historical evidence.

Reproduce from the project root (use your recorded RGB `.npy` for segmentation):

```bash
source ros2_ws/install-cuda/setup.bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
timeout -k 5s 90s .venv-ros-cuda/bin/python scripts/benchmark_icp_backends.py \
  --baseline cached --candidate cuda --output /tmp/cuda_icp.json
timeout -k 5s 90s .venv-ros-cuda/bin/python scripts/benchmark_segmentation_devices.py \
  --rgb output/cuda_benchmark_rgb.npy --output /tmp/cuda_segmentation.json
```

For live output timing, source the same install and run
`.venv-ros-cuda/bin/python scripts/measure_phase2.py --seconds 30 --output /tmp/cuda_live.json`.
Component benchmarks do not establish the final accepted-pose rate, physical pose
accuracy, or long-duration GPU stability. Those require a visible object and
representative motion/occlusion.

The first CUDA implementation was also verified with the live camera: ROS parameters
reported `cuda:0` for YOLO and `cuda` for ICP, and both processes held GPU memory.
A [10-second camera check](validation/cuda_live_pipeline_20260923.json) delivered
about 10.1 mask/cloud updates per second, with 17.1 ms median segmentation time
and 78.5 ms median receipt-to-cloud latency; all clouds in that window were empty.
A separate [15-second pose check](validation/cuda_live_poses_20260923.json) had
four insufficient-correspondence rejections and no accepted poses. These windows
verify operation and rejection handling, not sustained live tracking performance.

The revised GPU-resident pipeline was subsequently started headlessly with CUDA
YOLO and `icp_backend=cuda`. Both GPU processes were active; cached startup reached
`icp_ready` in about five seconds. The [10-second live check](validation/cuda_resident_live_20260923.json)
received 100 empty observations and no accepted poses. The optimized solver is
running, but this window does not establish a live accepted-pose rate.
