# Tracking performance (2026-09-23)

Follow-up: [CUDA implementation and CPU/GPU comparisons](CUDA_TRACKING.md).
CUDA substantially improves YOLO timing on the tested image. The revised
GPU-resident point-to-plane loop now beats cached CPU ICP on the supplied
reference in paired tests; the hybrid preset remains available.

The largest measured throughput problem is expensive unsuccessful ICP fits.
CPU segmentation is the next significant compute stage. Camera processing and
message delivery also contribute to end-to-end latency.

## Measurements before optimization

The [earlier live target run](validation/phase3_live_20260923.json) produced
70 accepted poses in 20 seconds (3.50 Hz), with 56 rejected fits:

| Work | Median | p95 |
| --- | ---: | ---: |
| Accepted ICP | 52.8 ms | 130.6 ms |
| Rejected ICP | 220.0 ms | 246.7 ms |
| ICP preparation, accepted frames | 5.1 ms | 9.4 ms |
| Camera receipt to accepted pose | 244.2 ms | 352.2 ms |

Rejected attempts typically consumed all 60 iterations. While the pose node is
busy, its bounded queue keeps only a pending observation; it cannot produce a
valid pose for every camera frame. The target output rate remains 10 Hz.

A subsequent [45-second no-target profile](validation/pipeline_idle_profile_20260923.json)
measured CPU segmentation at 64.7 ms median (131.9 ms p95), camera receipt to
publication at 43.4 ms median, and camera receipt to object-cloud publication at
159.9 ms median. Empty-cloud handling took 4.6 ms median. This run does **not**
measure populated-cloud filtering or ICP. These are separate windows and stage
medians must not be added as an exact per-frame latency decomposition. Receipt
starts in the camera worker, not at physical exposure.

## Implemented optimization

Previously each bounded ICP iteration called legacy Open3D `registration_icp`
with `max_iteration=1`. That call builds the target search tree and computes
initial and final correspondences on every invocation. See the
[Open3D 0.19 implementation](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/pipelines/registration/Registration.cpp).

`icp_workspace.py` now creates one CPU Float64 native nearest-neighbor index per
observed cloud, reuses the current correspondences for the next update, and
performs one batched search after each update. It uses Open3D's existing
point-to-point / point-to-plane transformation estimators. The fixed reference
stays immutable; each candidate is an absolute object-to-camera transform.

Rigid-transform checks, correspondence and observability checks, fitness/RMSE
thresholds, explicit pose-delta convergence, 60-iteration limit, and 250 ms budget
are unchanged. Faster execution can complete more work inside that budget, so
live outcomes near the time cutoff may differ. No additional dependencies or GPU
are required. An initial SciPy search prototype was slower and was discarded.

`icp_backend: cached` is the default in the shared Phase 2 configuration, inherited
by Phase 3. Set `icp_backend: open3d` in an ICP parameter override to use the old
loop. `/diagnostics` includes the selected backend. Parameters apply at launch.

## Controlled comparison

The [paired benchmark](validation/icp_backends_20260923.json) used your supplied
reference, 60 deterministic observations, two repeats, and alternating backend
execution order. Scenarios include complete warm-start observations, partial
observations with 0.7 mm noise and a near-pose seed, and the same partial clouds
with centroid initialization. Both backends receive identical clouds and seeds.
The benchmark uses a 2-second budget to compare equal work; production keeps
250 ms. The camera pipeline remained active during this CPU comparison.

| ICP solver timing | Original | Cached | Reduction |
| --- | ---: | ---: | ---: |
| Median | 13.43 ms | 9.03 ms | 33% |
| p95 | 158.28 ms | 77.89 ms | 51% |

Both versions accepted 106/120 attempts and rejected 14 for the iteration limit.
Acceptance, rejection reasons, and iteration counts matched. Maximum difference
between accepted poses was below 3e-16 m and 3e-15 rad. This checks equivalence,
not physical accuracy. See also the [CPU profile](validation/icp_backends_profile_20260923.txt).

The package builds and 70 tests pass. Tests cover point/plane backend equivalence
on full/partial noisy surfaces, input/reference immutability, search-index reuse,
invalid seeds/clouds, no correspondences, underconstrained planes, quality and
execution limits, previous-pose tracking, and ROS pose/TF/marker behavior.

Reproduce from the project root:

```bash
source ros2_ws/install/setup.bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-ros/bin/python \
  scripts/benchmark_icp_backends.py --output /tmp/icp_backends.json \
  --profile-output /tmp/icp_profile.txt
./scripts/test_phase1.sh -q
```

Synthetic solver speed is **not** an established live end-to-end speedup. There
was no target visible during the current camera profile. Repeat
`scripts/measure_phase2.py --seconds 30 --output /tmp/live_tracking.json` using
`.venv-ros/bin/python` with the target visible to assess accepted output rate,
rejections, and receipt-to-pose latency.

## Next optimization priorities

1. Measure with the actual target visible, including motion and partial views.
   Recovery that reduces long nonconvergent fits is likely more valuable than
   reducing the already-small cloud preparation time. Phase 4 loss/recovery work
   remains separate; this optimization does not implement that phase.
2. Benchmark segmentation input size or an alternative CPU inference engine on
   recorded RGB frames, checking mask/depth boundaries and pose accuracy. Smaller
   inputs trade segmentation detail for compute time; do not assume equal quality.
3. Profile camera-worker image rectification, packet copying, and publication
   scheduling to reduce the measured pre-segmentation latency.

Simply increasing camera rate or relaxing fit acceptance does not establish
faster valid pose tracking. Preserve the existing CPU defaults while evaluating
changes against the same recorded inputs.
