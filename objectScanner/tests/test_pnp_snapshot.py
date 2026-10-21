import json

import pytest

from scripts.pnp_snapshot import save, verify


def test_training_pnp_snapshot_survives_source_change(tmp_path):
    keypoints = tmp_path / "yolo_seg_pose_onthefly/object_keypoints.yaml"
    keypoints.parent.mkdir()
    keypoints.write_text("object_frame: box_center\n")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"revision": 2}))

    snapshot = save(tmp_path)
    assert save(tmp_path) == snapshot
    assert verify(snapshot)["calibration_revision"] == 2

    calibration.write_text(json.dumps({"revision": 3}))
    with pytest.raises(ValueError, match="differs from current"):
        save(tmp_path)
    assert json.loads((snapshot / "calibration.json").read_text())["revision"] == 2
    verify(snapshot)

    (snapshot / "object_keypoints.yaml").write_text("changed\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        verify(snapshot)
