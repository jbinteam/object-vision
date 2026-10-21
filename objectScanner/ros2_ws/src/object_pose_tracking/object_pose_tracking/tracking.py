"""Phase 3 seed memory only: a copy of the last accepted absolute raw ICP pose.

No observed cloud is stored, transformed, or used as the next source model.
Full tracking-state transitions and motion-jump rejection belong to Phase 4.
"""
import numpy as np
from .registration import is_rigid


class AcceptedPoseSeed:
    def __init__(self, timeout_seconds):
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('tracking_seed_timeout must be finite and positive')
        self.timeout_ns = int(timeout_seconds * 1e9)
        self.reset()

    def reset(self):
        self._pose = None
        self.stamp_ns = None

    def expire(self, now_ns):
        if self.stamp_ns is not None and not 0 <= now_ns - self.stamp_ns < self.timeout_ns:
            self.reset()
            return True
        return False

    def initial_guess(self, measurement_ns, now_ns):
        self.expire(now_ns)
        if self.stamp_ns is None:
            return None
        # Do not let a delayed/duplicate observation seed from a future pose.
        if measurement_ns <= self.stamp_ns:
            return None
        if measurement_ns - self.stamp_ns >= self.timeout_ns:
            self.reset()
            return None
        return self._pose.copy()  # backend mutation cannot corrupt the stored seed

    def accept(self, transform, measurement_ns, now_ns):
        if not is_rigid(transform):
            raise ValueError('Tracking seed must be a finite rigid transform')
        if not 0 <= now_ns - measurement_ns < self.timeout_ns:
            return False
        if self.stamp_ns is not None and measurement_ns <= self.stamp_ns:
            return False
        self._pose = np.asarray(transform, dtype=np.float64).copy()
        self._pose.setflags(write=False)
        self.stamp_ns = measurement_ns
        return True
