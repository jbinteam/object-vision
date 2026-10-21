"""Export objectScanner labels in NVlabs CenterPose's Objectron-style format."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from turntable_dataset import project, transform_points


# CenterPose/Objectron order: center, then the eight corners. CenterPose treats
# object Y as up; objectScanner treats object Z as up.
CENTERPOSE_UNIT_BOX = np.asarray([
    [0.0, 0.0, 0.0],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
], dtype=np.float64)

# p_objectScanner = OBJECTSCANNER_FROM_CENTERPOSE @ p_centerPose. This proper
# rotation maps CenterPose +Y (up) to objectScanner +Z and preserves handedness.
OBJECTSCANNER_FROM_CENTERPOSE = np.asarray([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# OpenCV optical coordinates (right, down, forward) to OpenGL camera
# coordinates (right, up, backward), as used by CenterPose's Objectron labels.
OPENGL_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def validate_category(value):
    if (not value or value != value.strip() or value in {".", ".."}
            or "/" in value or "\\" in value):
        raise ValueError("Category must be a single nonempty directory name")
    return value


def rotation_quaternion_xyzw(rotation):
    """Convert a proper 3x3 rotation matrix to a normalized XYZW quaternion."""
    rvec = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0][:, 0]
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    xyz = rvec / angle * np.sin(angle / 2.0)
    quaternion = np.r_[xyz, np.cos(angle / 2.0)]
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion.tolist()


def centerpose_projection_matrix(intrinsic, width, height, near=0.01, far=100.0):
    """Return the 4x4 projection used by CenterPose's Objectron augmentation."""
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    # CenterPose's loader converts the first NDC component to image Y and the
    # second to image X, so the first two rows deliberately appear swapped.
    return np.asarray([
        [0.0, -2.0 * fy / height, -(2.0 * cy / height - 1.0), 0.0],
        [2.0 * fx / width, 0.0, -(2.0 * cx / width - 1.0), 0.0],
        [0.0, 0.0, (far + near) / (near - far),
         2.0 * far * near / (near - far)],
        [0.0, 0.0, -1.0, 0.0],
    ])


def make_centerpose_annotation(label, category):
    """Convert one rich objectScanner label to CenterPose paired-JSON schema."""
    category = validate_category(category)
    intrinsic = np.asarray(label["intrinsic"], dtype=np.float64)
    camera_object = np.asarray(label["T_camera_object"], dtype=np.float64)
    dimensions = np.asarray(label["box_dimensions_m"], dtype=np.float64)
    width, height = (int(value) for value in label["image_size"])
    if intrinsic.shape != (3, 3) or camera_object.shape != (4, 4):
        raise ValueError("CenterPose export needs 3x3 intrinsics and a 4x4 pose")
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("CenterPose export needs three positive box dimensions")

    camera_centerpose = camera_object @ OBJECTSCANNER_FROM_CENTERPOSE
    # objectScanner dimensions are X/base-Y/up-Z; CenterPose dimensions are
    # X/up-Y/depth-Z.
    scale = dimensions[[0, 2, 1]]
    keypoints_local = CENTERPOSE_UNIT_BOX * scale
    keypoints_opencv = transform_points(camera_centerpose, keypoints_local)
    if np.any(keypoints_opencv[:, 2] <= 0):
        raise ValueError("CenterPose cuboid has a point behind the camera")
    projected = project(intrinsic, keypoints_opencv)

    camera_centerpose_gl = OPENGL_FROM_OPENCV @ camera_centerpose
    keypoints_gl = transform_points(camera_centerpose_gl, keypoints_local)
    projection = centerpose_projection_matrix(intrinsic, width, height)

    return {
        "camera_data": {
            "width": width,
            "height": height,
            "camera_view_matrix": np.eye(4).tolist(),
            "camera_projection_matrix": projection.tolist(),
            "location_world": [0.0, 0.0, 0.0],
            "quaternion_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "intrinsics": {
                "fx": float(intrinsic[0, 0]),
                "fy": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
            },
        },
        "objects": [{
            "class": category,
            "name": f"{category}_0",
            "provenance": "objectScanner",
            "location": camera_centerpose_gl[:3, 3].tolist(),
            "quaternion_xyzw": rotation_quaternion_xyzw(
                camera_centerpose_gl[:3, :3]),
            "projected_cuboid": projected.tolist(),
            "scale": scale.tolist(),
            "keypoints_3d": keypoints_gl.tolist(),
        }],
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare_centerpose_export(dataset, category, split):
    dataset = Path(dataset)
    category = validate_category(category)
    if split not in {"train", "val", "test"}:
        raise ValueError("CenterPose split must be train, val, or test")
    root = dataset / "centerpose"
    metadata_path = root / "export.json"
    metadata = {
        "schema_version": 1,
        "format": "NVlabs CenterPose Objectron-style paired PNG/JSON",
        "category": category,
        "split": split,
        "axis_conversion": "objectScanner Z-up to CenterPose Y-up",
        "source_dataset": str(dataset),
    }
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        for key in ("category", "split"):
            if saved.get(key) != metadata[key]:
                raise ValueError(
                    f"CenterPose export {key} is {saved.get(key)!r}, not {metadata[key]!r}")
    else:
        save_json(metadata_path, metadata)
    pair_dir = root / "outf" / f"{category}_{split}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    return pair_dir


def export_centerpose_sample(dataset, label, category, split="train"):
    """Write one same-basename PNG/JSON pair and return both destination paths."""
    dataset = Path(dataset)
    pair_dir = prepare_centerpose_export(dataset, category, split)
    image_source = dataset / label["image"]
    if not image_source.is_file():
        raise FileNotFoundError(f"Training image is missing: {image_source}")
    stem = Path(label["image"]).stem
    image_destination = pair_dir / f"{stem}{image_source.suffix.lower()}"
    label_destination = pair_dir / f"{stem}.json"
    shutil.copy2(image_source, image_destination)
    save_json(label_destination, make_centerpose_annotation(label, category))
    return image_destination, label_destination


def export_centerpose_dataset(dataset, category, split="train"):
    """Synchronize every recorded sample into a CenterPose training directory."""
    dataset = Path(dataset)
    count = 0
    pair_dir = prepare_centerpose_export(dataset, category, split)
    for label_path in sorted((dataset / "labels").glob("*.json")):
        export_centerpose_sample(dataset, json.loads(label_path.read_text()), category, split)
        count += 1
    return count, pair_dir
