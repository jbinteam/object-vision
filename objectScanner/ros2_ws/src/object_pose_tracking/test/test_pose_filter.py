import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from object_pose_tracking.pose_filter import PoseEMA, TrackingState


def transform(x=0., degrees=0.):
    value = np.eye(4)
    value[0, 3] = x
    value[:3, :3] = Rotation.from_euler('z', degrees, degrees=True).as_matrix()
    return value


def test_filter_ema_translation_and_slerp_rotation():
    filt = PoseEMA(.5, .5)
    np.testing.assert_allclose(filt.update(transform()), transform())
    output = filt.update(transform(2., 90.))
    assert output[0, 3] == pytest.approx(1.)
    assert Rotation.from_matrix(output[:3, :3]).magnitude() == pytest.approx(np.pi / 4)


def test_tracking_uses_previous_pose_and_loses_after_failures():
    state = TrackingState(2)
    assert state.needs_pnp
    state.accept(transform(.1))
    assert not state.needs_pnp and state.state == 'TRACKING'
    state.reject()
    assert not state.needs_pnp
    state.reject()
    assert state.needs_pnp and state.state == 'LOST'
