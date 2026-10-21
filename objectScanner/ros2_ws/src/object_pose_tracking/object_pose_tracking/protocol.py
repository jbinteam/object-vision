"""Bounded non-pickle RGB-D transport across the Python 3.10/3.12 boundary."""
import io
import struct
import numpy as np

MAX_PACKET = 32 * 1024 * 1024


def write_frame(stream, color, depth, intrinsic, timestamp_ns):
    payload = io.BytesIO()
    np.savez(payload, color=color, depth=depth, intrinsic=intrinsic,
             timestamp_ns=np.array(timestamp_ns, dtype=np.int64))
    data = payload.getvalue()
    if len(data) > MAX_PACKET:
        raise ValueError('Camera packet too large')
    stream.write(struct.pack('!I', len(data)))
    stream.write(data)
    stream.flush()


def read_exact(stream, count):
    pieces = bytearray()
    while len(pieces) < count:
        chunk = stream.read(count - len(pieces))
        if not chunk:
            raise EOFError('Camera worker closed its output')
        pieces.extend(chunk)
    return bytes(pieces)


def read_frame(stream):
    size, = struct.unpack('!I', read_exact(stream, 4))
    if not 0 < size <= MAX_PACKET:
        raise ValueError('Invalid camera packet size')
    with np.load(io.BytesIO(read_exact(stream, size)), allow_pickle=False) as data:
        color, depth, intrinsic = data['color'], data['depth'], data['intrinsic']
        timestamp = int(data['timestamp_ns'])
    if color.dtype != np.uint8 or depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError('Invalid RGB-D encoding')
    if color.shape != (*depth.shape, 3) or intrinsic.shape != (3, 3) or timestamp <= 0:
        raise ValueError('Invalid camera frame shape or timestamp')
    return color, depth, intrinsic, timestamp
