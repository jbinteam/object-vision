# Phase 4 performance review (2026-10-07)

This review concerns the current `scripts/collect_process/07_run_live.sh`
Phase 4 tracking path with the pinned `my_object` models. The older Phase 1
launcher defaults to CPU segmentation and a 10 Hz camera cap, with a different
`emergency_stop` checkpoint. Its shared cloud preprocessing receives the same
optimization, but the inference timings below do not measure that checkpoint.

Cloud voxel grouping and aggregation were the largest measured CPU stage on
eight saved target views. Packed integer voxel keys and weighted bincount sums
reduced complete cloud preprocessing from **8.55 ms to 3.79 ms median** (2.26x),
with p95 **11.20 ms to 4.91 ms**. Points, colors, voxel order and outlier-filter
results matched the original implementation exactly for every recorded input.
The packed code preserves XYZ lexical ordering; Python-integer range checks
prevent int64 overflow and retain the original grouping algorithm as fallback.

The [paired preprocessing report](validation/phase4_preprocessing_20261007.json)
uses eight evenly spaced saved L515 RGB-D/mask observations, three warm-ups and
20 timed repetitions per input with alternating baseline/current execution order.
It measures the complete voxelization plus statistical outlier filter, excluding
file loading and backprojection. The reference baseline is retained in the
benchmark script. These are recorded-input stage timings, not Gemini 305 live
throughput or physical pose accuracy measurements.

Reproduce from the project root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-ros-cuda/bin/python \
  scripts/benchmark_phase4_preprocessing.py \
  --output /tmp/phase4_preprocessing.json
```

The current [pinned-model inference comparison](validation/phase4_inference_20261007.json)
used four saved 960x540 L515 RGB images, three repetitions per device and unchanged
FP32 inference. All images produced a detection. CUDA segmentation measured
7.29 ms median versus CPU 36.33 ms; CUDA pose inference measured 6.92 ms versus
CPU 27.65 ms. The existing CUDA launch already selects GPU inference.
Tracking ordinarily runs segmentation alone; pose inference is reserved for
initialization/recovery. It now also skips the pose model when segmentation
selects no object, since those keypoints cannot be associated with a mask.

The current Phase 4 GICP solver uses the CPU legacy Open3D implementation even
with the CUDA launch. The [registration audit](validation/phase4_hotspots_20261007.json)
measured 4.57 ms median across changing recorded views under the existing
single-fine-iteration configuration. A source-normal cache improved repeated
static views but had only 1/32 hits and no speedup across changing views; it was
not added to runtime. No registration thresholds, visibility rules or iterations
were changed. Earlier Phase 3 CUDA ICP results concern a different solver.

Additional changes avoid optional work: Phase 4 pose publication creates Pose/TF
without unused RViz markers, and debug image/reference outputs render only when
they have subscribers. Required pose/TF and tracking-state publication continue
normally. Post-PnP and post-GICP freshness checks prevent expired computation
from publishing a new pose or changing the accepted seed/filter state.

The default camera publication cap remains **20 Hz**. A throughput experiment
can use `camera_rate:=30`, but a higher cap is not established faster valid
tracking. Live Gemini end-to-end latency, moving-target acceptance and sustained
output rate still need measurement. Camera/IPC changes were deferred: their
measured costs were smaller, and the existing pixel/unit/frame checks remain.

Validation passed: 159 ROS tests in the CPU environment (15 optional skips),
104 focused tests in the CUDA environment, and both CPU/CUDA ROS builds.
These include exact aggregation/fallback parity, real ROS pose/TF/marker
regressions, no-object recovery, subscriber-gated rendering and delayed
PnP/GICP expiry checks.
