"""Geometry contract for ICP reference creation. All internal coordinates are metres."""
import numpy as np


def vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return result


def rotation_from_degrees(degrees):
    """R_scan_obj = Rz(yaw) @ Ry(pitch) @ Rx(roll), using column vectors."""
    x, y, z = np.deg2rad(vector(degrees, "Rotation"))
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def rotation_to_degrees(rotation):
    """Inverse of ``rotation_from_degrees`` for Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r = np.asarray(rotation, dtype=np.float64)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6)):
        raise ValueError("Rotation must be an orthonormal right-handed 3x3 matrix")
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw])


def frame_from_points(origin, x_point, y_point):
    origin = vector(origin, "Origin")
    x = vector(x_point, "+X point") - origin
    y = vector(y_point, "+Y point") - origin
    if np.linalg.norm(x) < 1e-9 or np.linalg.norm(y) < 1e-9:
        raise ValueError("Axis points must differ from the origin")
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    y -= x * np.dot(x, y)
    if np.linalg.norm(y) < 1e-4:
        raise ValueError("Axis points must not be collinear")
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def crop_mask(points, minimum, maximum, rotation_deg=(0, 0, 0)):
    """Rotate the min/max box around its midpoint; test in box-local coordinates."""
    minimum, maximum = vector(minimum, "Crop minimum"), vector(maximum, "Crop maximum")
    if np.any(maximum <= minimum):
        raise ValueError("Every crop maximum must be greater than its minimum")
    rotation = rotation_from_degrees(rotation_deg)
    center = (minimum + maximum) / 2
    half = (maximum - minimum) / 2
    local = (np.asarray(points) - center) @ rotation
    # Include boundary points despite round-off in the inverse rotation.
    tolerance = 1e-12 * max(1., float(np.max(np.abs(center))), float(np.max(half)))
    return np.all(np.abs(local) <= half + tolerance, axis=1)


def frame_transforms(origin, rotation):
    origin = vector(origin, "Origin")
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Frame rotation must be a finite 3x3 matrix")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError("Frame must be orthonormal and right-handed")
    scan_obj = np.eye(4)
    scan_obj[:3, :3], scan_obj[:3, 3] = r, origin
    obj_scan = np.eye(4)
    obj_scan[:3, :3], obj_scan[:3, 3] = r.T, -r.T @ origin
    return scan_obj, obj_scan


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
