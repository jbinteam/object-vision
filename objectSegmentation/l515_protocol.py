"""Small bounded message protocol for the local Python 3.10 camera subprocess."""
import json
import select
import struct
import time

MAX_PAYLOAD = 64 * 1024 * 1024


def write_message(stream, metadata, payload=b""):
    header = json.dumps({**metadata, "payload_bytes": len(payload)}).encode()
    if len(header) > 65536 or len(payload) > MAX_PAYLOAD:
        raise ValueError("Camera message exceeds size limit")
    stream.write(struct.pack("<I", len(header)))
    stream.write(header)
    stream.write(payload)
    stream.flush()


def read_exact(stream, count, deadline):
    chunks = []
    while count:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            raise TimeoutError("Timed out waiting for the L515 camera process")
        chunk = stream.read(count)
        if not chunk:
            raise EOFError("L515 camera process closed the stream; check its error above")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def read_message(stream, timeout=15):
    deadline = time.monotonic() + timeout
    length = struct.unpack("<I", read_exact(stream, 4, deadline))[0]
    if not 0 < length <= 65536:
        raise ValueError("Invalid camera message header")
    header = json.loads(read_exact(stream, length, deadline))
    length = header.get("payload_bytes")
    if not isinstance(length, int) or not 0 <= length <= MAX_PAYLOAD:
        raise ValueError("Invalid camera payload length")
    return header, read_exact(stream, length, deadline)
