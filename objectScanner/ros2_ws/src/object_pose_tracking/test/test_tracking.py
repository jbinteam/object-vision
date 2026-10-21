import numpy as np
import pytest
from object_pose_tracking.tracking import AcceptedPoseSeed


def test_seed_is_last_accepted_absolute_pose_and_copied():
    state=AcceptedPoseSeed(.75)
    assert state.initial_guess(100,100) is None
    pose=np.eye(4);pose[:3,3]=[.1,.2,.7]
    expected=pose.copy()
    assert state.accept(pose,1_000_000_000,1_100_000_000)
    pose[:3,3]=9
    seed=state.initial_guess(1_200_000_000,1_250_000_000)
    np.testing.assert_array_equal(seed,expected)
    seed[:3,3]=8  # a backend may mutate its argument
    np.testing.assert_array_equal(state.initial_guess(1_300_000_000,1_350_000_000),expected)


def test_expiry_counts_from_last_acceptance_not_latest_attempt():
    state=AcceptedPoseSeed(.75)
    state.accept(np.eye(4),1_000_000_000,1_100_000_000)
    for stamp in (1_200_000_000,1_400_000_000,1_600_000_000):
        assert state.initial_guess(stamp,stamp+10_000_000) is not None
    assert state.initial_guess(1_750_000_000,1_750_000_000) is None
    assert state.stamp_ns is None


def test_stale_out_of_order_and_invalid_acceptance_cannot_replace_pose():
    state=AcceptedPoseSeed(.75)
    state.accept(np.eye(4),1_000_000_000,1_100_000_000)
    candidate=np.eye(4);candidate[0,3]=1
    assert not state.accept(candidate,900_000_000,1_200_000_000)
    assert not state.accept(candidate,1_100_000_000,2_000_000_000)
    bad=np.eye(4);bad[0,0]=2
    with pytest.raises(ValueError):state.accept(bad,1_100_000_000,1_200_000_000)
    np.testing.assert_array_equal(state.initial_guess(1_200_000_000,1_250_000_000),np.eye(4))
    assert state.initial_guess(1_000_000_000,1_200_000_000) is None
    assert state.expire(900_000_000)  # clock reset invalidates prior seed


def test_new_accepted_pose_advances_seed():
    state=AcceptedPoseSeed(.75)
    state.accept(np.eye(4),1_000_000_000,1_100_000_000)
    next_pose=np.eye(4);next_pose[1,3]=.1
    assert state.accept(next_pose,1_200_000_000,1_300_000_000)
    np.testing.assert_array_equal(state.initial_guess(1_800_000_000,1_850_000_000),next_pose)
    state.reset()
    assert state.initial_guess(1_900_000_000,1_900_000_000) is None
