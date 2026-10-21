#!/usr/bin/env python3
"""Local PLY cropping and obj_frame editor. Run: python webui.py"""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import uuid
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response
from werkzeug.exceptions import HTTPException

from reference_geometry import (crop_mask, frame_transforms, rotation_from_degrees,
                                rotation_to_degrees, transform_points)

ROOT = Path(__file__).resolve().parent
MAX_POINTS = 5_000_000
PREVIEW_POINTS = 120_000


def create_app(scan_dir=None, reference_dir=None, object_frame="obj_frame",
               fixed_frame_from_scan=False, production_reference_dir=None):
    app = Flask(__name__, static_folder=str(ROOT / "webui" / "dist"), static_url_path="/assets")
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
    scans = Path(scan_dir or ROOT / "output").resolve()
    references = Path(reference_dir or ROOT / "references").resolve()
    production = Path(production_reference_dir).resolve() if production_reference_dir else None
    if production:
        if not all((production / name).is_file() for name in
                   ("reference.ply", "frame.json", "object_keypoints.yaml")):
            raise ValueError("Production reference bundle is incomplete")
        frame = json.loads((production / "frame.json").read_text())
        if frame.get("method") != "calibrated_masked_rgbd_tsdf" or frame.get("frame_id") != "box_center":
            raise ValueError("Crop-only mode requires a calibrated box_center reference")
        scans = production
        references = production.parent
        object_frame = "box_center"
    if (not isinstance(object_frame, str) or not object_frame
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", object_frame)):
        raise ValueError("Object frame must start with a letter and use letters, numbers, _ or -")
    sessions = OrderedDict()
    lock = threading.RLock()

    @app.before_request
    def local_requests():
        if request.host.split(":")[0] not in ("127.0.0.1", "localhost", "[::1]"):
            return jsonify(error="Use localhost or 127.0.0.1"), 403
        if request.method == "POST" and request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
            return jsonify(error="Cross-origin requests are disabled"), 403

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, (ValueError, TypeError, KeyError)):
            return jsonify(error=str(exc)), 400
        if isinstance(exc, FileExistsError):
            return jsonify(error="That reference name already exists. Choose a new name."), 409
        app.logger.exception("Reference editor error")
        return jsonify(error="Could not process this cloud. See the server terminal for details."), 500

    def get_session(key):
        with lock:
            if key not in sessions:
                raise ValueError("Cloud session expired. Open the PLY again.")
            sessions.move_to_end(key)
            return sessions[key]

    def load(path, name, units):
        import open3d as o3d
        if units not in ("m", "mm"):
            raise ValueError("Choose metres or millimetres for source units")
        if fixed_frame_from_scan and units != "m":
            raise ValueError("Calibration-anchored scanner clouds are always in metres")
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        if len(points) < 3 or len(points) > MAX_POINTS:
            raise ValueError("PLY must contain between 3 and 5 million points")
        finite = np.isfinite(points).all(axis=1)
        cloud = cloud.select_by_index(np.flatnonzero(finite))
        points = np.asarray(cloud.points).copy() * (0.001 if units == "mm" else 1)
        if len(points) < 3:
            raise ValueError("PLY contains too few finite points")
        colors = np.asarray(cloud.colors).copy() if cloud.has_colors() else np.tile([0.68, 0.78, 0.84], (len(points), 1))
        colors = np.nan_to_num(colors, nan=0.7).clip(0, 1)
        normals = np.asarray(cloud.normals).copy() if cloud.has_normals() else None
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        # A flat source still needs adjustable bounds along all three axes.
        padding = np.maximum((maximum - minimum) * 0.005, 0.0001)
        ids = np.linspace(0, len(points) - 1, min(PREVIEW_POINTS, len(points)), dtype=np.int64)
        preview = np.column_stack([points[ids], colors[ids]]).astype("<f4").tobytes()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = uuid.uuid4().hex
        metadata = dict(id=key, name=name, points=len(points), preview_points=len(ids),
                        bounds_min=(minimum-padding).tolist(), bounds_max=(maximum+padding).tolist(),
                        extent=(maximum-minimum).tolist(), has_colors=cloud.has_colors(),
                        has_normals=normals is not None, source_units=units, units="m", sha256=digest)
        if production:
            metadata["suggested_frame"] = dict(frame_id="box_center", origin=[0, 0, 0],
                                                rotation_deg=[0, 0, 0], source=str(production))
        sidecar = path.with_name(path.stem + "_scan.json")
        if sidecar.is_file():
            scan_metadata = json.loads(sidecar.read_text())
            anchor = scan_metadata.get("reference_anchor")
            if anchor is not None:
                transform = np.asarray(anchor.get("T_scan_object"), dtype=np.float64)
                if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9)):
                    raise ValueError(f"Invalid reference anchor in {sidecar}")
                # Validate the rigid frame and expose the exact recipe to the browser.
                frame_transforms(transform[:3, 3], transform[:3, :3])
                metadata["suggested_frame"] = {
                    "frame_id": anchor.get("frame_id", object_frame),
                    "origin": transform[:3, 3].tolist(),
                    "rotation_deg": rotation_to_degrees(transform[:3, :3]).tolist(),
                    "source": str(sidecar),
                    "tag_reprojection_rms_px": anchor.get("tag_reprojection_rms_px"),
                }
                dimensions = np.asarray(anchor.get("box_dimensions_m"), dtype=np.float64)
                if dimensions.shape == (3,) and np.isfinite(dimensions).all() and np.all(dimensions > 0):
                    margin = 0.01
                    half = dimensions / 2 + margin
                    center = transform[:3, 3]
                    metadata["suggested_crop"] = {
                        "crop_min": (center - half).tolist(),
                        "crop_max": (center + half).tolist(),
                        "crop_rotation_deg": rotation_to_degrees(transform[:3, :3]).tolist(),
                        "box_dimensions_m": dimensions.tolist(),
                        "margin_m": margin,
                        "source": "capture calibration box_center dimensions",
                    }
        if fixed_frame_from_scan and "suggested_frame" not in metadata:
            raise ValueError(
                "This workflow requires a scan anchored by the capture calibration")
        with lock:
            sessions[key] = dict(points=points, colors=colors, normals=normals, preview=preview, metadata=metadata)
            while len(sessions) > 2:
                sessions.popitem(last=False)
        return metadata

    @app.get("/")
    def index():
        return send_from_directory(ROOT / "webui", "index.html")

    @app.get("/api/scans")
    def list_scans():
        paths = ([production / "reference.ply"] if production else
                 sorted(scans.glob("*.ply"), key=lambda p: p.stat().st_mtime, reverse=True))
        return jsonify([dict(name=p.name, bytes=p.stat().st_size) for p in paths
                        if not p.stem.endswith("_trajectory") and p.resolve().parent == scans])

    @app.post("/api/load")
    def load_local():
        body = request.get_json()
        name = body["name"]
        if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".ply"):
            raise ValueError("Choose a PLY from the scan folder")
        path = (scans / name).resolve()
        if path.parent != scans or not path.is_file() or (production and name != "reference.ply"):
            raise ValueError("Scan file not found")
        return jsonify(load(path, name, body.get("units", "m")))

    @app.post("/api/upload")
    def upload():
        if production:
            raise ValueError("Upload is disabled in production crop mode")
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename.lower().endswith(".ply"):
            raise ValueError("Upload a PLY file")
        with tempfile.TemporaryDirectory(prefix="reference-import-") as directory:
            path = Path(directory) / "cloud.ply"
            uploaded.save(path)
            return jsonify(load(path, Path(uploaded.filename).name, request.form.get("units", "m")))

    @app.get("/api/clouds/<key>/preview")
    def preview(key):
        return Response(get_session(key)["preview"], mimetype="application/octet-stream")

    @app.post("/api/count")
    def count():
        body = request.get_json()
        session = get_session(body["id"])
        mask = crop_mask(session["points"], body["crop_min"], body["crop_max"], body.get("crop_rotation_deg", [0, 0, 0]))
        return jsonify(points=int(mask.sum()))

    @app.post("/api/export")
    def export():
        import open3d as o3d
        body = request.get_json()
        name = body.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("Name must start with a letter or number; use only letters, numbers, _ and -")
        session = get_session(body["id"])
        crop_rotation = body.get("crop_rotation_deg", [0, 0, 0])
        mask = crop_mask(session["points"], body["crop_min"], body["crop_max"], crop_rotation)
        if production:
            if name != production.name:
                raise ValueError("Production reference name cannot change")
            if body.get("voxel_mm", 0) != 0:
                raise ValueError("Voxel downsampling is disabled in crop-only mode")
            count = int(mask.sum())
            if count < 1000 or count >= len(session["points"]):
                raise ValueError("Crop must remove points and retain at least 1000")
            source = production / "reference.ply"
            if hashlib.sha256(source.read_bytes()).hexdigest() != session["metadata"]["sha256"]:
                raise ValueError("Reference changed since it was opened; reload it")
            original_frame = json.loads((production / "frame.json").read_text())
            if original_frame.get("method") != "calibrated_masked_rgbd_tsdf" or original_frame.get("frame_id") != "box_center":
                raise ValueError("Production reference frame changed")
            cloud = o3d.io.read_point_cloud(str(source)).select_by_index(np.flatnonzero(mask))
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            backup = production.parent / f".{production.name}-before-crop-{timestamp}"
            stage = Path(tempfile.mkdtemp(prefix=f".{production.name}-crop-", dir=production.parent))
            try:
                for path in production.iterdir():
                    if path.is_file() and path.name != "reference_bundle.zip":
                        shutil.copy2(path, stage / path.name)
                if not o3d.io.write_point_cloud(str(stage / "reference.ply"), cloud):
                    raise RuntimeError("PLY crop export failed")
                original_frame["points"] = count
                original_frame.setdefault("manual_crops", []).append(dict(
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source_points=len(session["points"]), kept_points=count,
                    crop_min=body["crop_min"], crop_max=body["crop_max"],
                    crop_rotation_deg=crop_rotation, backup=str(backup)))
                (stage / "frame.json").write_text(json.dumps(original_frame, indent=2) + "\n")
                with zipfile.ZipFile(stage / "reference_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
                    for filename in ("reference.ply", "frame.json", "object_keypoints.yaml", "README.txt"):
                        bundle.write(stage / filename, arcname=f"{name}/{filename}")
                production.rename(backup)
                try:
                    stage.rename(production)
                except Exception:
                    backup.rename(production)
                    raise
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            return jsonify(name=name, points=count, directory=str(production), backup=str(backup),
                           downloads={kind: f"/api/references/{name}/{file}" for kind, file in
                                      (("ply", "reference.ply"), ("frame", "frame.json"),
                                       ("bundle", "reference_bundle.zip"))})
        if mask.sum() < 3:
            raise ValueError("Keep at least 3 points before exporting")
        fixed = session["metadata"].get("suggested_frame") if fixed_frame_from_scan else None
        origin = fixed["origin"] if fixed else body["origin"]
        rotation_deg = fixed["rotation_deg"] if fixed else body["rotation_deg"]
        rotation = rotation_from_degrees(rotation_deg)
        scan_obj, obj_scan = frame_transforms(origin, rotation)
        voxel = float(body.get("voxel_mm", 0))
        if not np.isfinite(voxel) or not 0 <= voxel <= 1000:
            raise ValueError("Voxel size must be between 0 and 1000 mm")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(transform_points(session["points"][mask], obj_scan))
        cloud.colors = o3d.utility.Vector3dVector(session["colors"][mask])
        if session["normals"] is not None:
            normals = session["normals"][mask] @ rotation
            if np.isfinite(normals).all():
                cloud.normals = o3d.utility.Vector3dVector(normals)
        if voxel > 0:
            cloud = cloud.voxel_down_sample(voxel / 1000)
        if len(cloud.points) < 3:
            raise ValueError("Voxel size leaves fewer than 3 points; reduce it")
        if body.get("estimate_normals", True):
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=min(30, len(cloud.points))))
            # Orient towards the original scan camera origin in object coordinates.
            cloud.orient_normals_towards_camera_location(obj_scan[:3, 3])
        metadata = dict(schema_version=2, name=name, frame_id=object_frame, units="m",
                        source=session["metadata"], points_before_downsample=int(mask.sum()),
                        points=len(cloud.points), has_normals=cloud.has_normals(),
                        T_scan_obj=scan_obj.tolist(), T_obj_scan=obj_scan.tolist(),
                        convention="Column vectors: p_scan = T_scan_obj @ p_obj; p_obj = T_obj_scan @ p_scan",
                        rotation_convention="R_scan_obj = Rz(yaw) @ Ry(pitch) @ Rx(roll); degrees",
                        crop_convention="Rotate the crop_min/crop_max box about its midpoint in scan coordinates; R_scan_crop = Rz @ Ry @ Rx from crop_rotation_deg (roll, pitch, yaw). Dimensions are crop_max - crop_min.",
                        frame_source=("capture calibration embedded in scan metadata"
                                      if fixed else "manual reference editor"),
                        recipe={"crop_min": body["crop_min"], "crop_max": body["crop_max"],
                                "origin": origin, "rotation_deg": rotation_deg,
                                "crop_rotation_deg": crop_rotation},
                        voxel_mm=voxel, normals_orientation="towards original scan origin" if body.get("estimate_normals", True) else "source normals, if present")
        references.mkdir(parents=True, exist_ok=True)
        destination = references / name
        if destination.exists():
            raise FileExistsError(name)
        temporary = Path(tempfile.mkdtemp(prefix=".reference-", dir=references))
        try:
            if not o3d.io.write_point_cloud(str(temporary / "reference.ply"), cloud):
                raise RuntimeError("PLY export failed")
            (temporary / "frame.json").write_text(json.dumps(metadata, indent=2) + "\n")
            (temporary / "README.txt").write_text(
                f"reference.ply is a cropped reference in {object_frame}, in metres.\n"
                "frame.json stores both inverse transforms and the crop recipe.\n"
                "For ICP, use reference.ply as source and a live scene cloud as target.\n"
                "The resulting source-to-target transform is T_scene_obj.\n"
                "ICP needs an adequate initial pose and matching units; it is not global object detection.\n"
                "The crop was applied in the source scan frame before transformation.\n"
                "Re-estimated normals face the original scan origin; inspect for multi-view objects.\n")
            with zipfile.ZipFile(temporary / "reference_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
                for filename in ("reference.ply", "frame.json", "README.txt"):
                    bundle.write(temporary / filename, arcname=f"{name}/{filename}")
            with lock:
                if destination.exists():
                    raise FileExistsError(name)
                temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return jsonify(name=name, points=len(cloud.points), directory=str(destination),
                       downloads={kind: f"/api/references/{name}/{file}" for kind, file in
                                  (("ply", "reference.ply"), ("frame", "frame.json"), ("bundle", "reference_bundle.zip"))})

    @app.get("/api/references/<name>/<filename>")
    def download(name, filename):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or filename not in ("reference.ply", "frame.json", "reference_bundle.zip"):
            raise ValueError("Invalid reference file")
        return send_from_directory(references / name, filename, as_attachment=True)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scan-dir", type=Path,
                        help="Directory containing raw PLY scans (default: output/)")
    parser.add_argument("--reference-dir", type=Path,
                        help="Directory receiving reference bundles (default: references/)")
    parser.add_argument("--object-frame", default="obj_frame",
                        help="Frame ID recorded in frame.json (default: obj_frame)")
    parser.add_argument("--fixed-frame-from-scan", action="store_true",
                        help="Require and preserve the calibration anchor embedded in *_scan.json")
    parser.add_argument("--production-reference-dir", type=Path,
                        help="Crop an existing calibrated reference without changing its object frame")
    args = parser.parse_args()
    suffix = ("/?frame_id=box_center&lock_frame=1&crop_only=1"
              if args.production_reference_dir else
              f"/?frame_id={args.object_frame}&lock_frame=1" if args.fixed_frame_from_scan else "")
    print(f"Reference editor: http://127.0.0.1:{args.port}{suffix}", flush=True)
    create_app(args.scan_dir, args.reference_dir, args.object_frame,
               args.fixed_frame_from_scan, args.production_reference_dir).run(
        host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
