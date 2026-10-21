"""Timestamped transport for ordered 2D keypoints using standard PointCloud2."""
import numpy as np
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2


FIELDS = [PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
          for index, name in enumerate(('id', 'u', 'v', 'confidence', 'valid'))]


def create_keypoints(header, keypoints):
    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('keypoints must be N x 4: u,v,confidence,valid')
    packed = np.column_stack((np.arange(len(points), dtype=np.float32), points))
    return point_cloud2.create_cloud(header, FIELDS, packed)


def read_keypoints(message, expected_count=None):
    rows = list(point_cloud2.read_points(message,
                                         field_names=('id', 'u', 'v', 'confidence', 'valid'),
                                         skip_nans=False))
    if expected_count is None:
        expected_count = len(rows)
    output = np.full((expected_count, 4), [np.nan, np.nan, 0., 0.], dtype=np.float64)
    seen = set()
    for row in rows:
        index = int(row[0])
        if index in seen or not 0 <= index < expected_count:
            raise ValueError('Duplicate or out-of-range keypoint ID')
        output[index] = [row[1], row[2], row[3], row[4]]
        seen.add(index)
    return output
