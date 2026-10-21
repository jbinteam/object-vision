"""Dense NOCS target generation from aligned metric depth and object poses."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


UINT16_MAX = np.iinfo(np.uint16).max


def _validated_inputs(depth_mm, mask, label):
    depth_mm = np.asarray(depth_mm)
    mask = np.asarray(mask)
    if depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
        raise ValueError("Depth must be a single-channel uint16 millimetre image")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != depth_mm.shape:
        raise ValueError("Mask and aligned depth must have identical dimensions")

    width, height = label["image_size"]
    if depth_mm.shape != (height, width):
        raise ValueError(
            f"Depth shape {depth_mm.shape} does not match label image_size {(width, height)}")
    intrinsic = np.asarray(label["intrinsic"], dtype=np.float64)
    camera_object = np.asarray(label["T_camera_object"], dtype=np.float64)
    dimensions = np.asarray(label["box_dimensions_m"], dtype=np.float64)
    if intrinsic.shape != (3, 3) or camera_object.shape != (4, 4):
        raise ValueError("NOCS labels require 3x3 intrinsics and a 4x4 T_camera_object")
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("NOCS labels require three positive metric box dimensions")
    if not np.isfinite(intrinsic).all() or not np.isfinite(camera_object).all():
        raise ValueError("NOCS calibration contains non-finite values")
    rotation = camera_object[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("T_camera_object rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("T_camera_object rotation is not right-handed")
    depth_unit_m = float(label.get("depth_unit_m", 0.001))
    if not np.isfinite(depth_unit_m) or depth_unit_m <= 0:
        raise ValueError("depth_unit_m must be a positive finite metric scale")
    alignment = label.get("depth_aligned_to")
    if alignment is not None and alignment != "rectified color image":
        raise ValueError("Depth must be aligned to the rectified color image")
    return (depth_mm, mask > 0, intrinsic, camera_object, dimensions,
            depth_unit_m)


def extract_masked_object_points(depth_mm, mask, label, box_tolerance_ratio=0.08):
    """Deproject masked depth and return matching camera/object-frame points.

    ``T_camera_object`` follows ``P_camera = T_camera_object @ P_object``.
    Returned object points therefore use the calibrated box-center origin and
    axes directly; no later reference-axis assignment is required.
    """
    if not 0 <= box_tolerance_ratio <= 0.5:
        raise ValueError("box_tolerance_ratio must be between 0 and 0.5")
    (depth_mm, foreground, intrinsic, camera_object, dimensions,
     depth_unit_m) = _validated_inputs(depth_mm, mask, label)

    depth_valid = foreground & (depth_mm > 0)
    ys, xs = np.nonzero(depth_valid)
    if len(xs):
        z = depth_mm[ys, xs].astype(np.float64) * depth_unit_m
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        camera_points = np.column_stack(((xs - cx) * z / fx,
                                         (ys - cy) * z / fy,
                                         z))
        rotation = camera_object[:3, :3]
        translation = camera_object[:3, 3]
        object_points = (camera_points - translation) @ rotation
        tolerance = np.maximum(0.003, dimensions * box_tolerance_ratio)
        inside = np.all(np.abs(object_points) <= dimensions / 2 + tolerance, axis=1)
        xs_valid, ys_valid = xs[inside], ys[inside]
        camera_valid = camera_points[inside]
        object_valid = object_points[inside]
    else:
        xs_valid = ys_valid = np.empty(0, dtype=np.int64)
        camera_valid = np.empty((0, 3), dtype=np.float64)
        object_valid = np.empty((0, 3), dtype=np.float64)

    diagonal = float(np.linalg.norm(dimensions))
    normalized = object_valid / diagonal

    if len(object_valid):
        roundtrip_camera = object_valid @ camera_object[:3, :3].T + camera_object[:3, 3]
        projected_x = intrinsic[0, 0] * roundtrip_camera[:, 0] / roundtrip_camera[:, 2] + intrinsic[0, 2]
        projected_y = intrinsic[1, 1] * roundtrip_camera[:, 1] / roundtrip_camera[:, 2] + intrinsic[1, 2]
        reprojection = np.hypot(projected_x - xs_valid, projected_y - ys_valid)
        reprojection_max = float(reprojection.max())
    else:
        reprojection_max = None
    mask_pixels = int(np.count_nonzero(foreground))
    depth_pixels = int(np.count_nonzero(depth_valid))
    valid_pixels = int(len(object_valid))
    metrics = {
        "mask_pixels": mask_pixels,
        "masked_depth_pixels": depth_pixels,
        "valid_nocs_pixels": valid_pixels,
        "depth_coverage": depth_pixels / mask_pixels if mask_pixels else 0.0,
        "nocs_coverage": valid_pixels / mask_pixels if mask_pixels else 0.0,
        "box_consistency": valid_pixels / depth_pixels if depth_pixels else 0.0,
        "max_reprojection_error_px": reprojection_max,
        "box_diagonal_m": diagonal,
        "depth_unit_m": depth_unit_m,
        "coordinate_min": normalized.min(axis=0).tolist() if len(normalized) else None,
        "coordinate_max": normalized.max(axis=0).tolist() if len(normalized) else None,
    }
    return foreground, ys_valid, xs_valid, camera_valid, object_valid, metrics


def generate_nocs_target(depth_mm, mask, label, box_tolerance_ratio=0.08):
    """Return an instance map, logical RGBA NOCS map, and validation metrics.

    The first three logical channels contain box-centered object coordinates
    normalized by the 3D box diagonal and shifted by +0.5 for unsigned PNG
    storage. The fourth channel is the valid-coordinate mask. Instance pixels
    without usable depth retain instance ID 1 but have an invalid NOCS channel.
    """
    (foreground, ys_valid, xs_valid, _camera_valid,
     object_valid, metrics) = extract_masked_object_points(
        depth_mm, mask, label, box_tolerance_ratio)
    instance = np.zeros(foreground.shape, dtype=np.uint16)
    instance[foreground] = 1
    rgba = np.zeros((*foreground.shape, 4), dtype=np.uint16)
    diagonal = metrics["box_diagonal_m"]
    packed = object_valid / diagonal + 0.5
    if len(packed) and (np.any(packed < 0) or np.any(packed > 1)):
        raise ValueError("Box-filtered NOCS coordinates exceed the uint16 packing range")
    encoded = np.rint(np.clip(packed, 0, 1) * UINT16_MAX).astype(np.uint16)
    rgba[ys_valid, xs_valid, :3] = encoded
    rgba[ys_valid, xs_valid, 3] = UINT16_MAX
    return instance, rgba, metrics


def write_nocs_png(path, rgba):
    """Write logical XYZ-valid RGBA channels as a standards-compliant 16-bit PNG."""
    path = Path(path)
    rgba = np.asarray(rgba)
    if rgba.dtype != np.uint16 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("NOCS PNG data must be uint16 RGBA")
    path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV's PNG encoder expects BGRA. Reorder so non-OpenCV readers see XYZ+A.
    bgra = rgba[..., [2, 1, 0, 3]]
    if not cv2.imwrite(str(path), bgra):
        raise OSError(f"Cannot write NOCS PNG: {path}")


def read_nocs_png(path):
    """Read a generator NOCS PNG into logical XYZ-valid RGBA channel order."""
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.dtype != np.uint16 or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"Not a uint16 four-channel NOCS PNG: {path}")
    return bgra[..., [2, 1, 0, 3]]
