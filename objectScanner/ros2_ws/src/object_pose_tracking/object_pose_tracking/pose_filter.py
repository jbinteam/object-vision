"""Validation-aware pose smoothing; invalid poses must never enter this state."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .registration import is_rigid


class PoseEMA:
    def __init__(self, translation_alpha=.45, rotation_alpha=.45):
        if not 0 < translation_alpha <= 1 or not 0 < rotation_alpha <= 1:
            raise ValueError('Pose filter alphas must be in (0,1]')
        self.translation_alpha = translation_alpha
        self.rotation_alpha = rotation_alpha
        self.value = None

    def reset(self):
        self.value = None

    def update(self, transform):
        transform = np.asarray(transform, dtype=float)
        if not is_rigid(transform):
            raise ValueError('Only a validated rigid pose can be filtered')
        if self.value is None:
            self.value = transform.copy()
            return self.value.copy()
        output = np.eye(4)
        a = self.translation_alpha
        output[:3, 3] = (1 - a) * self.value[:3, 3] + a * transform[:3, 3]
        rotations = Rotation.from_matrix(np.stack((self.value[:3, :3], transform[:3, :3])))
        output[:3, :3] = Slerp([0., 1.], rotations)(self.rotation_alpha).as_matrix()
        self.value = output
        return output.copy()


class TrackingState:
    INITIALIZATION = 'INITIALIZATION'
    TRACKING = 'TRACKING'
    LOST = 'LOST'

    def __init__(self, failures_until_lost=2):
        if failures_until_lost < 1:
            raise ValueError('failures_until_lost must be positive')
        self.failures_until_lost = failures_until_lost
        self.state = self.INITIALIZATION
        self.pose = None
        self.failures = 0

    @property
    def needs_pnp(self):
        return self.state != self.TRACKING or self.pose is None

    def accept(self, pose):
        if not is_rigid(pose):
            raise ValueError('Cannot track a nonrigid pose')
        self.pose = np.asarray(pose).copy()
        self.failures = 0
        self.state = self.TRACKING

    def reject(self):
        self.failures += 1
        if self.failures >= self.failures_until_lost:
            self.state, self.pose = self.LOST, None

    def reset(self):
        self.state, self.pose, self.failures = self.LOST, None, 0
