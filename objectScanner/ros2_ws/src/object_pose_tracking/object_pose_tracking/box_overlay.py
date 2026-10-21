"""Draw the calibrated eight-corner cuboid on an RGB camera image."""

import cv2
import numpy as np


# Corner IDs are ordered by z, then y, then x in object_keypoints.yaml.
CUBOID_EDGES = tuple((index, index ^ (1 << axis))
                     for index in range(8) for axis in range(3)
                     if index < (index ^ (1 << axis)))


def draw_box_wireframe(rgb, projected_corners=None):
    """Draw all 12 cuboid edges; return the source image when no pose is accepted."""
    image = rgb.copy()
    if projected_corners is None:
        cv2.putText(image, 'NO ACCEPTED POSE', (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (255, 80, 80), 2, cv2.LINE_AA)
        return image
    points = np.asarray(projected_corners, dtype=np.float64)
    if points.shape != (8, 2) or not np.isfinite(points).all():
        raise ValueError('Wireframe needs eight finite projected corners')
    corners = np.rint(np.clip(points, -1e6, 1e6)).astype(np.int32)
    for first, second in CUBOID_EDGES:
        cv2.line(image, tuple(corners[first]), tuple(corners[second]),
                 (0, 255, 0), 2, cv2.LINE_AA)
    for index, point in enumerate(corners):
        cv2.circle(image, tuple(point), 3, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(image, str(index), tuple(point + [4, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, 'GICP 3D box', (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                .7, (0, 255, 0), 2, cv2.LINE_AA)
    return image
