#!/usr/bin/env python3
"""Scan a stationary object with a moving RGB-D camera into a colored PLY."""
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time

# Avoid CPU oversubscription on machines with many logical cores.
os.environ.setdefault("OMP_NUM_THREADS", "4")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--camera-backend", choices=("orbbec", "realsense", "auto"), default="orbbec",
                   help="RGB-D backend (default: Orbbec Gemini 305)")
    p.add_argument("--serial", help="Camera serial number (needed if multiple cameras are connected)")
    p.add_argument("--anchor-calibration", type=Path,
                   help="AprilTag calibration.json used to anchor scan_frame to box_center")
    p.add_argument("-s", "--seconds", type=float, default=30)
    naming = p.add_mutually_exclusive_group()
    naming.add_argument("-o", "--output", type=Path, help="PLY path; default: output/scan_TIMESTAMP.ply")
    naming.add_argument("-n", "--name", help="Scan name (e.g. mug or mug.ply); saves into output/")
    p.add_argument("--voxel", type=float, default=3, help="TSDF voxel size in mm (default: 3)")
    p.add_argument("--depth-min", type=float, default=0.25, help="Near distance in metres")
    p.add_argument("--depth-max", "--depth-trunc", type=float, default=1.5, help="Far distance in metres")
    p.add_argument("--fov-keep", type=float, default=0.9, help="Central fraction per image axis")
    p.add_argument("--downscale", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu",
                   help="Compute device (default: cpu; CUDA must be explicitly selected)")
    p.add_argument("--safe", action="store_true",
                   help="CPU only, no windows, at least 2x downscale and 5 mm voxels")
    p.add_argument("--stall-timeout", type=float, default=15,
                   help="Kill worker if a processing operation stalls this many seconds (default: 15)")
    p.add_argument("--block-count", type=int, default=10000)
    p.add_argument("--min-fitness", type=float, default=0.1,
                   help="Minimum tracking fitness (default: 0.1); raise for stricter rejection")
    p.add_argument("--isolate", action="store_true", help="Remove dominant plane; keep largest DBSCAN cluster")
    p.add_argument("--plane-thresh", type=float, default=8, help="Plane threshold in mm")
    p.add_argument("--cluster-eps", type=float, default=15, help="DBSCAN radius in mm")
    p.add_argument("--min-points", type=int, default=50)
    p.add_argument("--mesh", action="store_true", help="Also export a triangle mesh")
    p.add_argument("--no-view", action="store_true", help="Disable both preview windows")
    preview = p.add_mutually_exclusive_group()
    preview.add_argument("--no-3d", dest="no_3d", action="store_true", default=True,
                         help="Show only the RGB/depth preview (default)")
    preview.add_argument("--3d", dest="no_3d", action="store_false",
                         help="Also enable the live fused 3D preview")
    p.add_argument("--preview-every", type=int, default=15, help="Update fused 3D preview every N tracked frames")
    a = p.parse_args(argv)
    import math
    if not math.isfinite(a.stall_timeout) or a.stall_timeout <= 0:
        p.error("--stall-timeout must be finite and positive")
    for name in ("seconds", "voxel", "depth_min", "depth_max", "fov_keep", "min_fitness", "plane_thresh", "cluster_eps"):
        if not math.isfinite(getattr(a, name)):
            p.error(f"--{name.replace('_', '-')} must be finite")
    if a.seconds <= 0 or a.voxel <= 0:
        p.error("duration and voxel size must be positive")
    if not 0 < a.depth_min < a.depth_max <= 65:
        p.error("require 0 < depth-min < depth-max <= 65 metres")
    if not 0.1 < a.fov_keep <= 1 or not 0 <= a.min_fitness <= 1:
        p.error("fov-keep must be in (0.1, 1]; min-fitness must be in [0, 1]")
    if a.block_count < 1000 or a.preview_every < 1:
        p.error("block-count must be >= 1000 and preview-every >= 1")
    if a.plane_thresh <= 0 or a.cluster_eps <= 0 or a.min_points < 1:
        p.error("isolation thresholds and min-points must be positive")
    if a.output and a.output.suffix.lower() != ".ply":
        p.error("--output must end in .ply")
    if a.anchor_calibration is not None:
        a.anchor_calibration = a.anchor_calibration.expanduser().resolve()
        if not a.anchor_calibration.is_file():
            p.error("--anchor-calibration must be an existing calibration.json")
    if a.name is not None:
        name = a.name.strip()
        stem = name[:-4] if name.lower().endswith(".ply") else name
        if not stem or stem in (".", "..") or any(c in name for c in ("/", chr(92), chr(0))):
            p.error("--name must be a filename without directories; use --output for a path")
        a.output = Path(__file__).resolve().parent / "output" / (stem + ".ply")
    if a.safe:
        a.device = "cpu"
        a.no_view = True
        a.downscale = max(a.downscale, 2)
        a.voxel = max(a.voxel, 5)
    return a


def output_paths(args):
    output = args.output or Path(__file__).resolve().parent / "output" / (
        "scan_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".ply")
    paths = {"cloud": output, "metadata": output.with_name(output.stem + "_scan.json"),
             "log": output.with_name(output.stem + "_run.log"),
             "trajectory": output.with_name(output.stem + "_trajectory.ply")}
    if args.mesh:
        paths["mesh"] = output.with_name(output.stem + "_mesh.ply")
    for path in paths.values():
        if path.exists():
            raise RuntimeError(f"Output already exists: {path}. Choose another --output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    return paths


def save_scan(slam, intrinsic, args, paths, camera_info, reference_anchor=None):
    import numpy as np
    import open3d as o3d
    from reconstruction import _isolate_object_cloud, _crop_mesh_to_object

    cloud = slam.extract_point_cloud()
    if not len(cloud.points):
        raise RuntimeError("No surface reconstructed. Keep the object within range and move slowly.")
    if args.isolate:
        candidate, info = _isolate_object_cloud(cloud, True, args.plane_thresh / 1000,
                                                args.cluster_eps / 1000, args.min_points)
        print(f"Isolation: {info}")
        if len(candidate.points):
            cloud = candidate
    if not o3d.io.write_point_cloud(str(paths["cloud"]), cloud):
        raise RuntimeError(f"Failed to save {paths['cloud']}")
    poses = np.asarray(slam.trajectory)
    trajectory = o3d.geometry.PointCloud()
    trajectory.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
    trajectory.paint_uniform_color([1, 0.4, 0])
    if not o3d.io.write_point_cloud(str(paths["trajectory"]), trajectory):
        raise RuntimeError("Could not save camera trajectory")
    metadata = {"units": "metres", "pose_convention": "T_world_camera; first camera frame is world origin; x right, y down, z forward",
                "camera": camera_info, "intrinsic": intrinsic.tolist(),
                "poses": poses.tolist(), "tracked_frames": slam.tracked_frames,
                "lost_frames": slam.lost_frames, "points": len(cloud.points),
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    if reference_anchor is not None:
        metadata["reference_anchor"] = reference_anchor
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(cloud.points):,} colored points: {paths['cloud']}")
    if args.mesh:
        mesh = slam.extract_mesh()
        if args.isolate:
            mesh = _crop_mesh_to_object(mesh, cloud, margin_m=0.01)
        if not len(mesh.triangles) or not o3d.io.write_triangle_mesh(str(paths["mesh"]), mesh):
            raise RuntimeError("Point cloud saved, but mesh extraction/export failed")
        print(f"Saved mesh: {paths['mesh']}")


def scan(args, stop=None):
    import open3d as o3d

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    use_cuda = args.device != "cpu" and o3d.core.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA requested but unavailable in Open3D. Use --device cpu.")
    device = o3d.core.Device("CUDA:0" if args.device != "cpu" and use_cuda else "CPU:0")
    paths = output_paths(args)
    handler = logging.FileHandler(paths["log"])
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("object_scanner")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        _capture(args, paths, device, logger, stop=stop)
    except Exception:
        logger.exception("Scan failed")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


def _acquire_reference_anchor(camera, args, calibration, detector, title, stop=None):
    """Return the exact first reconstruction frame and its calibrated object pose."""
    import cv2
    import numpy as np
    from turntable_dataset import detect_tags, make_label

    print("Hold the camera still with at least two mapped permanent tags visible. "
          "Scanning starts after box_center anchoring succeeds.", flush=True)
    while True:
        if stop is not None and stop.is_set():
            raise RuntimeError("Reference scan stopped before box_center anchoring")
        frame = camera.read(args.depth_min, args.depth_max, args.fov_keep, args.downscale)
        if frame is None:
            continue
        detections = detect_tags(frame.color, detector)
        try:
            label = make_label(
                calibration, detections, frame.intrinsic, frame.color.shape,
                frame.timestamp_ns or time.time_ns())
            anchor = {
                "frame_id": label["object_frame"],
                "T_scan_object": label["T_camera_object"],
                "convention": "p_scan = T_scan_object @ p_object; scan_frame is the first reconstructed camera frame",
                "permanent_tags_used": label["permanent_tags_used"],
                "tag_reprojection_rms_px": label["tag_reprojection_rms_px"],
                "calibration": str(args.anchor_calibration),
                "calibration_revision": label["calibration_revision"],
                "box_dimensions_m": label["box_dimensions_m"],
            }
            print(f"Anchor acquired from {len(label['permanent_tags_used'])} tags; "
                  f"error {label['tag_reprojection_rms_px']:.2f} px. Start moving the camera.",
                  flush=True)
            return anchor, frame
        except (ValueError, cv2.error) as exc:
            if args.no_view:
                continue
            image = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
            for tag_id, corners in detections.items():
                cv2.polylines(image, [np.rint(corners).astype(np.int32)], True,
                              (0, 220, 255), 2)
                center = tuple(np.rint(corners.mean(axis=0)).astype(int))
                cv2.putText(image, str(tag_id), center, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
            message = f"WAITING FOR BOX ANCHOR: {exc}"
            cv2.putText(image, message[:100], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(image, message[:100], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            cv2.imshow(title, image)
            if cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')):
                raise RuntimeError("Reference scan cancelled before box_center anchoring")


def _capture(args, paths, device, logger, stop=None):
    import cv2
    import numpy as np
    import open3d as o3d
    from camera import open_camera
    from reconstruction import TensorSlam
    from viewer import CloudPreview

    calibration = None
    detector = None
    if args.anchor_calibration is not None:
        from turntable_dataset import make_detector
        calibration = json.loads(args.anchor_calibration.read_text())
        detector = make_detector(calibration["tag_family"])

    logger.info("Starting device=%s settings=%s", device, vars(args))
    logger.info("Camera startup")
    camera = open_camera(args.camera_backend, args.serial)
    if calibration and (
            calibration.get("camera_serial") not in (None, camera.info["serial"])
            or calibration.get("camera_backend") not in (None, camera.info.get("backend"))):
        camera.close()
        raise RuntimeError("Anchor calibration was created with a different camera")
    print(f"{camera.info['name']} | compute {device} | {args.seconds:g}s | voxel {args.voxel:g} mm")
    print("Keep the object stationary and move the camera slowly around it.", flush=True)
    print("Ctrl+C stops; Q/ESC saves and D toggles depth when preview windows are enabled.", flush=True)
    slam = None
    preview = None
    intrinsic = None
    overlay = True
    failure = None
    model_usable = True
    reference_anchor = None
    pending_frame = None
    title = "RGB-D Object Scanner | Q saves | D depth"
    try:
        if not args.no_view:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            if not args.no_3d:
                preview = CloudPreview("RGB-D fused 3D cloud | Q saves")
        if calibration is not None:
            reference_anchor, pending_frame = _acquire_reference_anchor(
                camera, args, calibration, detector, title, stop)
        start = time.monotonic()
        report = start
        while time.monotonic() - start < args.seconds:
            if stop is not None and stop.is_set():
                break
            logger.info("Read aligned camera frame")
            frame = pending_frame
            pending_frame = None
            if frame is None:
                frame = camera.read(args.depth_min, args.depth_max, args.fov_keep, args.downscale)
            if frame is None:
                continue
            if slam is None:
                h, w = frame.depth.shape
                intrinsic = frame.intrinsic
                logger.info("Initialize reconstruction %dx%d", w, h)
                slam = TensorSlam(o3d.core.Tensor(intrinsic), w, h, args.voxel,
                                  args.depth_max, device, args.block_count, args.min_fitness)
                slam.depth_min = args.depth_min
            try:
                logger.info("Process frame accepted=%d lost=%d", slam.tracked_frames, slam.lost_frames)
                tracked = slam.process(frame.color, frame.depth)
                if preview and tracked and (slam.tracked_frames == 1 or slam.tracked_frames % args.preview_every == 0):
                    logger.info("Extract live preview")
                    cloud = slam.extract_point_cloud()
            except RuntimeError:
                model_usable = False
                raise
            if preview:
                if tracked and (slam.tracked_frames == 1 or slam.tracked_frames % args.preview_every == 0):
                    logger.info("Update live 3D window")
                    preview.update(cloud)
                if not preview.poll():
                    break
            now = time.monotonic()
            if now - report >= 1:
                print(f"{now-start:5.1f}s  tracked={slam.tracked_frames} lost={slam.lost_frames} fitness={slam.last_fitness:.2f}", flush=True)
                report = now
            if not args.no_view:
                image = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
                if overlay:
                    valid = frame.depth > 0
                    norm = np.clip(1 - frame.depth.astype(float) / (args.depth_max * 1000), 0, 1)
                    tint = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    blend = cv2.addWeighted(image, 0.5, tint, 0.5, 0)
                    image[valid] = blend[valid]
                status = "TRACKING" if tracked else "LOST - return to last view"
                text = f"{status} | {args.seconds-(now-start):.0f}s | frames {slam.tracked_frames}"
                cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(title, image)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('d'), ord('D')):
                    overlay = not overlay
    except KeyboardInterrupt:
        print("Stopping and saving captured data.")
    except RuntimeError as exc:
        failure = exc
        logger.exception("Capture failed; model_usable=%s", model_usable)
        print(f"Capture stopped: {exc}", file=sys.stderr)
    finally:
        logger.info("Close camera and preview windows")
        actions = [camera.close]
        if preview:
            actions.append(preview.close)
        if not args.no_view:
            actions.append(cv2.destroyAllWindows)
        for close in actions:
            try:
                close()
            except Exception:
                logger.exception("Resource cleanup failed")
    if not model_usable:
        print(f"Reconstruction failed; skipping extraction/export. Diagnostic log: {paths['log']}", file=sys.stderr)
    elif slam is not None and slam.tracked_frames:
        logger.info("Extract and save final scan")
        save_scan(slam, intrinsic, args, paths, camera.info, reference_anchor)
        logger.info("Scan saved")
    elif failure is None:
        raise RuntimeError("No valid frames captured; nothing saved.")
    if failure:
        raise failure


def main():
    import faulthandler
    faulthandler.enable()
    args = arguments()
    try:
        if args.list_devices:
            from camera import devices
            print(json.dumps(devices(args.camera_backend), indent=2))
        else:
            from supervisor import run_scan
            return run_scan(args)
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install requirements.txt in the Python 3.10/3.11 environment.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
