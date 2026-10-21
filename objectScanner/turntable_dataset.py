"""AprilTag turntable geometry and label generation (metres, RGB optical frame)."""

from __future__ import annotations

import cv2
import numpy as np


EDGES = ((0, 1), (1, 3), (3, 2), (2, 0),
         (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))
TAG_FAMILIES = ("tag36h11", "tagStandard41h12")


class _PupilDetectorAdapter:
    def __init__(self, detector):
        self.detector = detector

    def detectMarkers(self, gray):
        detections = self.detector.detect(gray)
        # pupil-apriltags reports BL, BR, TR, TL for tagStandard41h12;
        # the geometry below uses TL, TR, BR, BL, as OpenCV ArUco does.
        corners = [det.corners[[3, 2, 1, 0]][None, :, :].astype(np.float32)
                   for det in detections]
        ids = (np.array([[det.tag_id] for det in detections], dtype=np.int32)
               if detections else None)
        return corners, ids, []


def make_detector(family):
    if family == "tag36h11":
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    if family == "tagStandard41h12":
        try:
            from pupil_apriltags import Detector
        except ImportError as exc:
            raise RuntimeError("tagStandard41h12 needs pupil-apriltags in this environment") from exc
        return _PupilDetectorAdapter(Detector(
            families=family, nthreads=4, quad_decimate=1.0, quad_sigma=0.0,
            refine_edges=1, decode_sharpening=0.25))
    raise ValueError(f"Unsupported AprilTag family: {family}")


def inverse(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def transform_points(transform, points):
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def tag_corners(size):
    half = size / 2
    # AprilTag detector order: top-left, top-right, bottom-right, bottom-left.
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], dtype=np.float64)


def pose_from_correspondences(object_points, image_points, intrinsic, *, square=False):
    object_points = np.ascontiguousarray(object_points, dtype=np.float64)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64)
    method = cv2.SOLVEPNP_IPPE_SQUARE if square else cv2.SOLVEPNP_SQPNP
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, intrinsic, None, flags=method)
    if not ok or tvec[2, 0] <= 0:
        raise ValueError("AprilTag pose is behind the camera")
    rotation = cv2.Rodrigues(rvec)[0]
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec[:, 0]
    projected = project(intrinsic, transform_points(transform, object_points))
    error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
    return transform, error


def project(intrinsic, camera_points):
    points = np.asarray(camera_points, dtype=np.float64)
    if np.any(points[:, 2] <= 0):
        raise ValueError("Box lies behind the camera")
    return np.column_stack((intrinsic[0, 0] * points[:, 0] / points[:, 2] + intrinsic[0, 2],
                            intrinsic[1, 1] * points[:, 1] / points[:, 2] + intrinsic[1, 2]))


def detect_tags(rgb, detector):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(tag_id): np.asarray(corner[0], dtype=np.float64)
            for corner, tag_id in zip(corners, ids[:, 0])}


def remove_tags(rgb, detections, tag_ids, margin_px=6):
    """Inpaint known fiducials for the training input; preserve raw RGB separately."""
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for tag_id in tag_ids:
        if tag_id in detections:
            cv2.fillConvexPoly(mask, np.rint(detections[tag_id]).astype(np.int32), 255)
    if not np.any(mask):
        return rgb.copy()
    radius = max(1, 2 * int(margin_px) + 1)
    mask = cv2.dilate(mask, np.ones((radius, radius), dtype=np.uint8))
    return cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)


def individual_poses(detections, tag_ids, size, intrinsic, max_error_px):
    poses = {}
    for tag_id in tag_ids:
        if tag_id not in detections:
            continue
        try:
            pose, error = pose_from_correspondences(
                tag_corners(size), detections[tag_id], intrinsic, square=True)
            if error <= max_error_px:
                poses[tag_id] = pose
        except (ValueError, cv2.error):
            pass
    return poses


def sort_setup_tags(poses, setup_ids, camera_table):
    """Resolve unordered setup tags into base origin, base X/Y corners, and top."""
    table_camera = inverse(camera_table)
    table_centers = {
        tag_id: transform_points(table_camera, poses[tag_id][:3, 3][None, :])[0]
        for tag_id in setup_ids
    }
    top_id = max(setup_ids, key=lambda tag_id: abs(table_centers[tag_id][2]))
    floor_ids = [tag_id for tag_id in setup_ids if tag_id != top_id]
    floor_error = max(abs(table_centers[tag_id][2]) for tag_id in floor_ids)
    top_height = abs(table_centers[top_id][2])
    if floor_error > .03:
        raise ValueError("Three setup tags must lie on the turntable surface")
    if top_height - floor_error <= .01:
        raise ValueError("One setup tag must be clearly above the three surface tags")

    centers = {tag_id: poses[tag_id][:3, 3] for tag_id in setup_ids}
    candidates = []
    for origin_id in floor_ids:
        adjacent = [tag_id for tag_id in floor_ids if tag_id != origin_id]
        first = centers[adjacent[0]] - centers[origin_id]
        second = centers[adjacent[1]] - centers[origin_id]
        lengths = np.linalg.norm(first), np.linalg.norm(second)
        if min(lengths) <= .01:
            continue
        cosine = abs(float(np.dot(first, second) / (lengths[0] * lengths[1])))
        candidates.append((cosine, origin_id, adjacent[0], adjacent[1]))
    if not candidates:
        raise ValueError("Surface setup tags are too close together")
    cosine, origin_id, x_id, y_id = min(candidates)
    if cosine > .35:
        raise ValueError("Three surface tags must mark three corners of a rectangular base")

    p0, px, py, ptop = [centers[tag_id] for tag_id in (origin_id, x_id, y_id, top_id)]
    if np.dot(np.cross(px - p0, py - p0), ptop - p0) < 0:
        x_id, y_id = y_id, x_id
    return {"base_origin": origin_id, "base_x": x_id,
            "base_y": y_id, "top": top_id}


def resolve_box_geometry(poses, setup_ids, camera_table):
    roles = sort_setup_tags(poses, setup_ids, camera_table)
    p0, p1, p2, ptop = [poses[roles[role]][:3, 3]
                         for role in ("base_origin", "base_x", "base_y", "top")]
    x = p1 - p0
    width = float(np.linalg.norm(x))
    if width <= .01:
        raise ValueError("Bottom X markers are too close")
    x /= width
    y = p2 - p0
    y -= x * np.dot(y, x)
    length = float(np.linalg.norm(y))
    if length <= .01:
        raise ValueError("Bottom Y marker is too close to the X axis")
    y /= length
    z = np.cross(x, y)
    height = float(np.dot(ptop - p0, z))
    if min(width, length, height) <= .01:
        raise ValueError("Box dimensions must each exceed 1 cm; check tag placement")
    if max(width, length, height) > 2:
        raise ValueError("Box dimension exceeds 2 m; check tag size and placement")
    camera_object = np.eye(4)
    camera_object[:3, :3] = np.column_stack((x, y, z))
    camera_object[:3, 3] = p0 + x * width / 2 + y * length / 2 + z * height / 2
    return roles, [width, length, height], camera_object


def calibrate(detections, intrinsic, setup_ids, permanent_ids, setup_tag_size,
              permanent_tag_size=None, max_error_px=2.5, tag_family="tag36h11"):
    permanent_tag_size = (setup_tag_size if permanent_tag_size is None
                          else permanent_tag_size)
    if tag_family not in TAG_FAMILIES:
        raise ValueError(f"Unsupported AprilTag family: {tag_family}")
    if len(set(setup_ids + permanent_ids)) != len(setup_ids + permanent_ids):
        raise ValueError("Setup and permanent tag IDs must be distinct")
    setup_poses = individual_poses(detections, setup_ids, setup_tag_size,
                                   intrinsic, max_error_px)
    permanent_poses = individual_poses(detections, permanent_ids, permanent_tag_size,
                                       intrinsic, max_error_px)
    poses = {**setup_poses, **permanent_poses}
    missing = [tag_id for tag_id in setup_ids if tag_id not in poses]
    if missing:
        raise ValueError(f"Setup tags missing or inaccurate: {missing}")
    visible = [tag_id for tag_id in permanent_ids if tag_id in poses]
    if len(visible) < 2:
        raise ValueError("Need at least two visible permanent tags to set the box")

    # Freeze visible permanent tag corners in the first visible tag's frame.
    origin_id = visible[0]
    camera_to_table = inverse(poses[origin_id])
    board = {str(tag_id): (camera_to_table @ poses[tag_id]).tolist() for tag_id in visible}
    board[str(origin_id)] = np.eye(4).tolist()
    for tag_id in visible[1:]:
        table_tag = np.asarray(board[str(tag_id)])
        if abs(table_tag[2, 3]) > .015 or table_tag[2, 2] < .94:
            raise ValueError(f"Permanent tag {tag_id} is not coplanar with tag {origin_id}")
    camera_table, board_error = board_pose(detections, intrinsic, board, permanent_tag_size,
                                           max_error_px=max_error_px)

    roles, dimensions, camera_object = resolve_box_geometry(poses, setup_ids, camera_table)
    table_object = inverse(camera_table) @ camera_object
    return {
        "schema_version": 3,
        "tag_family": tag_family,
        "setup_tag_size_m": setup_tag_size,
        "permanent_tag_size_m": permanent_tag_size,
        "setup_ids": sorted(setup_ids),
        "resolved_setup_roles": roles,
        "permanent_ids": sorted(permanent_ids),
        "table_frame": f"permanent_tag_{origin_id}",
        "object_frame": "box_center",
        "box_dimensions_m": dimensions,
        "T_table_object": table_object.tolist(),
        "T_table_tags": board,
        "calibration_reprojection_rms_px": board_error,
        "revision": 1,
    }


def reconfigure_box(calibration, detections, intrinsic, max_error_px=2.5):
    """Resolve a new box while preserving the calibrated permanent-tag map."""
    setup_size = calibration.get("setup_tag_size_m", calibration.get("tag_size_m"))
    permanent_size = calibration.get("permanent_tag_size_m", calibration.get("tag_size_m"))
    if setup_size is None or permanent_size is None:
        raise ValueError("Calibration is missing tag sizes")
    setup_ids = calibration["setup_ids"]
    poses = individual_poses(detections, setup_ids, setup_size, intrinsic, max_error_px)
    missing = [tag_id for tag_id in setup_ids if tag_id not in poses]
    if missing:
        raise ValueError(f"Setup tags missing or inaccurate: {missing}")
    camera_table, error = board_pose(detections, intrinsic, calibration["T_table_tags"],
                                     permanent_size, max_error_px)
    roles, dimensions, camera_object = resolve_box_geometry(poses, setup_ids, camera_table)
    updated = {**calibration}
    updated.update(
        schema_version=max(3, calibration.get("schema_version", 1)),
        resolved_setup_roles=roles,
        box_dimensions_m=dimensions,
        T_table_object=(inverse(camera_table) @ camera_object).tolist(),
        calibration_reprojection_rms_px=error,
        revision=calibration.get("revision", 1) + 1,
    )
    return updated


def register_visible_tags(calibration, detections, intrinsic, max_error_px=2.5):
    """Extend a fixed table map using new tags seen with two already mapped tags."""
    board = calibration["T_table_tags"].copy()
    tag_size = calibration.get("permanent_tag_size_m", calibration.get("tag_size_m"))
    if tag_size is None:
        raise ValueError("Calibration is missing permanent tag size")
    camera_table, _ = board_pose(detections, intrinsic, board,
                                 tag_size, max_error_px)
    candidates = [tag_id for tag_id in calibration["permanent_ids"]
                  if str(tag_id) not in board and tag_id in detections]
    poses = individual_poses(detections, candidates, tag_size,
                             intrinsic, max_error_px)
    added = []
    for tag_id in candidates:
        if tag_id not in poses:
            continue
        table_tag = inverse(camera_table) @ poses[tag_id]
        # All permanent tags are mounted on the same table plane, face up.
        if abs(table_tag[2, 3]) > .015 or table_tag[2, 2] < .94:
            continue
        trial = {**board, str(tag_id): table_tag.tolist()}
        try:
            board_pose(detections, intrinsic, trial, tag_size,
                       max_error_px=max_error_px)
        except (ValueError, cv2.error):
            continue
        board = trial
        added.append(tag_id)
    updated = {**calibration, "T_table_tags": board}
    return updated, added


def board_pose(detections, intrinsic, board, tag_size, max_error_px=3.0):
    visible = [tag_id for tag_id in board if int(tag_id) in detections]
    if len(visible) < 2:
        raise ValueError("At least two permanent tags must be visible")
    corners = tag_corners(tag_size)
    object_points = np.concatenate([
        transform_points(np.asarray(board[tag_id]), corners) for tag_id in visible])
    image_points = np.concatenate([detections[int(tag_id)] for tag_id in visible])
    transform, error = pose_from_correspondences(object_points, image_points, intrinsic)
    if error > max_error_px:
        raise ValueError(f"Permanent tag reprojection error {error:.2f} px exceeds {max_error_px:.2f} px")
    return transform, error


def box_corners(dimensions):
    hx, hy, hz = np.asarray(dimensions, dtype=np.float64) / 2
    return np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz],
                     [-hx, -hy, hz], [hx, -hy, hz], [-hx, hy, hz], [hx, hy, hz]])


def pose_label_fields(calibration, camera_object, intrinsic, image_shape):
    points_camera = transform_points(camera_object, box_corners(calibration["box_dimensions_m"]))
    uv = project(intrinsic, points_camera)
    center = project(intrinsic, camera_object[:3, 3][None, :])[0]
    height, width = image_shape[:2]
    clipped = [max(0, float(uv[:, 0].min())), max(0, float(uv[:, 1].min())),
               min(width - 1, float(uv[:, 0].max())), min(height - 1, float(uv[:, 1].max()))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("Object box is outside the image")
    rvec = cv2.Rodrigues(camera_object[:3, :3])[0][:, 0]
    return {
        "T_camera_object": camera_object.tolist(),
        "center_xyz_m": camera_object[:3, 3].tolist(),
        "rotation_vector_rad": rvec.tolist(),
        "center_uv_px": center.tolist(),
        "box_dimensions_m": list(calibration["box_dimensions_m"]),
        "box_corners_uv_px": uv.tolist(),
        "bbox_xyxy_px_clipped": clipped,
        "bbox_partly_outside_image": bool(np.any(uv[:, 0] < 0) or np.any(uv[:, 0] >= width)
                                          or np.any(uv[:, 1] < 0) or np.any(uv[:, 1] >= height)),
        "calibration_revision": calibration.get("revision", 1),
    }


def make_label(calibration, detections, intrinsic, image_shape, timestamp_ns,
               max_error_px=3.0):
    tag_size = calibration.get("permanent_tag_size_m", calibration.get("tag_size_m"))
    if tag_size is None:
        raise ValueError("Calibration is missing permanent tag size")
    camera_table, error = board_pose(detections, intrinsic, calibration["T_table_tags"],
                                     tag_size, max_error_px)
    camera_object = camera_table @ np.asarray(calibration["T_table_object"])
    height, width = image_shape[:2]
    return {
        "timestamp_ns": int(timestamp_ns),
        "camera_frame": "camera_color_optical_frame",
        "object_frame": "box_center",
        "units": "m",
        "intrinsic": intrinsic.tolist(),
        "image_size": [width, height],
        **pose_label_fields(calibration, camera_object, intrinsic, image_shape),
        "permanent_tags_used": [int(key) for key in calibration["T_table_tags"]
                                if int(key) in detections],
        "tag_reprojection_rms_px": error,
    }


def relabel_for_reconfigured_box(label, old_calibration, new_calibration):
    """Move an existing label to a new table-fixed object box definition."""
    old_camera_object = np.asarray(label["T_camera_object"], dtype=np.float64)
    old_table_object = np.asarray(old_calibration["T_table_object"], dtype=np.float64)
    camera_table = old_camera_object @ inverse(old_table_object)
    camera_object = camera_table @ np.asarray(new_calibration["T_table_object"], dtype=np.float64)
    intrinsic = np.asarray(label["intrinsic"], dtype=np.float64)
    width, height = label["image_size"]
    return {
        **label,
        **pose_label_fields(new_calibration, camera_object, intrinsic, (height, width)),
    }
