# Orbbec Gemini 305 camera

The scanner, turntable/NOCS capture tools, AprilTag camera test, and ROS RGB-D
worker select Gemini 305 with the `orbbec` backend by default. The existing L515
backend remains available explicitly. Object geometry, reference frames,
keypoint ordering, training checkpoints, and saved datasets are not migrated.

## SDK and installation

Use the existing `.venv` (Python 3.10 on this Linux x86-64 machine):

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-orbbec.txt
.venv/bin/python scanner.py --list-devices --camera-backend orbbec
```

The pinned distribution is **`pyorbbecsdk2==2.1.2`**, imported as `pyorbbecsdk`.
The [official Python release](https://github.com/orbbec/pyorbbecsdk/releases/tag/v2.1.2)
embeds **Orbbec SDK v2.9.3** and includes Linux x86-64 Python 3.10 and 3.12 wheels.
This is distinct from separately installed newer C++ SDK or ROS driver versions.
Do not install the older `pyorbbecsdk` distribution in place of this v2 wrapper.

Connect the camera to a direct USB 3 port and close other camera applications.
The connected device on this machine enumerates as **`2bc5:0840` Gemini 305**
at USB 5000 Mbit/s. The existing
`/etc/udev/rules.d/99-obsensor-libusb.rules` covers IDs `0840` and `0841`.
For another machine, use the
[official SDK udev rules](https://github.com/orbbec/pyorbbecsdk/blob/v2.1.2/scripts/env_setup/99-obsensor-libusb.rules).
The migration does not update device firmware.

Enumeration does not start streams. If multiple supported cameras are attached,
specify `--serial SERIAL`; ROS launch uses `camera_serial:=SERIAL`.
`--camera-backend auto` discovers installed supported backends and refuses
ambiguity. Explicit `orbbec` never falls back to L515 on an SDK or stream error.

## Capture and ROS launch

```bash
.venv/bin/python scanner.py --camera-backend orbbec --serial SERIAL --safe
.venv/bin/python scripts/test_apriltags.py --camera --camera-backend orbbec
./scripts/build_phase1.sh
./scripts/run_phase1.sh camera_backend:=orbbec camera_serial:=SERIAL rviz:=true
```

Phases 2, 3, 3 CUDA, and 4 expose the same backend/serial launch inputs. Keep
`camera_python` pointed to `.venv/bin/python`; the SDK runs in the camera worker,
isolated from ROS/inference Python 3.12. Existing `camera:=false` launches still
accept external aligned RGB-D topics.

For L515:

```bash
.venv/bin/python -m pip install -r requirements-realsense.txt
.venv/bin/python scanner.py --camera-backend realsense
./scripts/run_phase1.sh camera_backend:=realsense
```

The legacy AprilTag `--l515` option still selects the RealSense backend.

## Closest-range mode

The Gemini 305 adapter now requires the device-advertised **Close Range Default**
preset before creating its pipeline or enumerating stream profiles. It checks the
mode-switch status and mode readback, selects disparity search range **256**, and
reads the property back. The pinned
[SDK v2.9.3 property definition](https://github.com/orbbec/OrbbecSDK_v2/blob/v2.9.3/include/libobsensor/h/Property.h)
maps `OB_PROP_DISP_SEARCH_RANGE_MODE_INT` value **2** to 256 (value 1 means 128).
The adapter also verifies disparity offset 0, the observed default of this preset
on firmware 1.0.70. An unavailable preset, unsupported property or mismatched
readback fails startup rather than silently streaming another mode. Verified
settings appear in camera metadata and the camera worker's stderr log.

Depth profile fallback is limited to **640x480, 848x480 or 848x530**. These are the
resolutions with a specified **40 mm minimum Z at disparity 256** in Table 4-8-1 of
the [manufacturer Gemini 305 datasheet, V1.0](https://orbbec-debian-repos-aws.s3.amazonaws.com/product/Orbbec_Gemini%20305%20Datasheet%20V1.0_20260105.pdf).
The datasheet specifies 50 mm for 1280x720/800 and says 256 is supported only in
Close-Range presets. The adapter continues to prefer 640x480 at 30 FPS and retains
software alignment to rectified RGB calibration.

Phase 4 detection's camera and object-cloud software minimum-depth filters are
**0.04 m**; their upper filter stays **1.5 m**. The live launch retains
`camera_rate:=30`. The camera worker's default software filters in other capture
workflows are set by their own arguments and can discard nearby depth. The turntable
and NOCS collectors now also default to **0.04–1.5 m** with `orbbec`, and forward
that interval to optional diagnostic scans. Explicit `realsense` or `auto`
collection retains the 0.25 m near cutoff; `--depth-min` overrides either default. The hardware
preset itself is selected whenever this adapter opens the Gemini 305.

Verified 2026-10-07 on serial `CV27864000C8`, firmware 1.0.70: actual streaming
readback was Close Range Default, disparity property 2 and offset 0. A bounded
sample delivered 30 consecutive 640x480 RGB/depth pairs in 1.00146 s
(29.96 FPS); timestamp skew was 0 us, depth remained uint16 millimetres and all
alignment/calibration guards passed. The adapter then closed cleanly. All 30
targeted offline camera tests passed, including rejected mode/property readback,
unavailable 256 search range and exclusion of profiles without the 40 mm specification.
This verifies mode selection and streaming. Metric depth accuracy and reliable
object recognition or pose tracking at 4 cm require a separate physical measurement.

## Camera and geometry contract

Both backends produce contiguous RGB `uint8`, depth `uint16` **millimetres**,
and the 3x3 intrinsic matrix for the output's **rectified color optical frame**:
X right, Y down, Z forward. ROS retains the established
`camera_color_optical_frame` name, `rgb8`/`16UC1` messages, zero distortion
CameraInfo, and shared host wall-clock receipt timestamp. This frame name
describes the camera convention; it is not a calibrated transform from the old
physical camera. Any external camera-to-robot transform must be recalibrated.

Gemini depth scale is reported in millimetres per raw count and converted to
metres for the common masking helper
([SDK frame API](https://github.com/orbbec/pyorbbecsdk/blob/v2.1.2/sdk/include/libobsensor/hpp/Frame.hpp)). SDK alignment targets the color frame;
the default Orbbec v2 AlignFilter rectifies depth while preserving raw color.
The adapter rectifies RGB once and verifies that aligned depth calibration
matches it
([SDK v2.9.3 alignment implementation](https://github.com/orbbec/OrbbecSDK_v2/blob/v2.9.3/src/filter/publicfilters/Align.cpp)). It rejects incompatible calibration rather than publishing zero
distortion for unverified pixels. Downscaling updates focal lengths and principal
points using the existing pixel-center convention.

## New camera, new capture session

An L515 calibration session is tied to its camera serial and cannot be resumed
with Gemini 305. Existing calibration files stay intact and mismatches fail.
Create a new dataset name and perform the normal AprilTag setup/calibration:

```bash
scripts/collect_process/01_capture.sh my_object_gemini305 my_object --serial SERIAL
```

Without arguments, all [collection steps](../scripts/collect_process/README.md)
now use `my_object_gemini305` and category `my_object`; steps 1 and 7 select
Orbbec explicitly. Choose another session for a changed physical setup.

New calibrations record camera serial, backend and model; every capture retains
the measured frame intrinsics. Use the existing collection workflow to create
a new reference only when requested. Existing pinned reference/keypoint snapshots
remain valid object-frame assets for runtime use; changing the camera alone does
not authorize relabeling or retraining them. Image appearance and field of view
can differ, so verify detection, PnP reprojection and depth alignment on the new
camera before accepting live tracking accuracy.

## Validation scope

Validated on 2026-10-07 with Gemini 305 serial `CV27864000C8`, firmware
`1.0.70`, USB 3.2, and the pinned Python SDK:

- Actual enumeration, profile queries, and AlignFilter configuration passed.
- A bounded camera smoke delivered 30 consecutive 640x480 RGB/depth pairs at
  approximately 30.19 FPS during the sample. Runtime checks accepted aligned
  intrinsics, zero depth distortion, and identity depth-to-color extrinsics.
  Last device timestamp skew was 1 microsecond. In the test scene,
  96.23% of depth pixels were nonzero using a 0.1–5 m interval; median depth
  was 937 mm. Camera startup, warm-up and smoke took 5.40 seconds and closed
  cleanly. These scene values do not establish metric accuracy.
- The actual Python 3.10 camera worker sent 10 binary packets to the Python 3.12
  ROS environment. With `--downscale 2`, RGB was `(240, 320, 3)` uint8 and depth
  `(240, 320)` uint16 millimetres. Scaled intrinsics were
  `fx=184.8588257, fy=184.7225037, cx=160.3172455, cy=120.3048172`.
  Timestamps increased, the last packet was approximately 4 ms old, and SDK
  logs stayed on stderr. SIGINT ran camera cleanup and exited without a forced
  kill. The default 0.15–1.5 m mask retained 70.73% nonzero pixels in that scene.
- Offline checks passed: 124 scanner unit tests, 135 ROS tests (15 skipped),
  a separate pinned-snapshot test, CPU/CUDA ROS builds, and all six launch
  helpers' argument inspection. SDK-free camera/worker/snapshot checks in the
  ROS CUDA environment passed 37 tests with four optional-SDK skips.

Offline tests include actual installed SDK structure bindings, format conversion,
metric scaling and saturation, alignment/intrinsics guards, timestamp skew drops
and recovery, startup cleanup, binary pipe isolation, and old-session preservation.
The real image and pipe checks used in-memory frames only. Physical metric depth
accuracy, moving-edge registration, actual ROS camera-topic streaming, sustained
rates, and full Gemini pose tracking remain unverified. Previous L515 tracking
results do not establish those Gemini acceptance criteria.
