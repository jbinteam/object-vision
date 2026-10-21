"""PointCloud2 XYZ + packed RGB compatible with RViz's RGB8 transformer."""
import numpy as np
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2


def create_colored_cloud(header, points, colors):
    points, colors = np.asarray(points), np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape or colors.dtype != np.uint8:
        raise ValueError('Expected N x 3 XYZ and matching uint8 RGB')
    fields = [PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
              for name, offset in [('x', 0), ('y', 4), ('z', 8)]]
    fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
    data = np.empty(len(points), dtype=point_cloud2.dtype_from_fields(fields))
    for i, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, i]
    rgb = colors.astype(np.uint32)
    data['rgb'] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    return point_cloud2.create_cloud(header, fields, data)
