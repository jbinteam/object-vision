#!/usr/bin/env python3
"""Pin the calibrated PnP inputs at the start of model training."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


FILES = {
    "calibration.json": Path("calibration.json"),
    "object_keypoints.yaml": Path("yolo_seg_pose_onthefly/object_keypoints.yaml"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(snapshot):
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("sha256", {})) != set(FILES):
        raise ValueError("Invalid PnP snapshot manifest")
    for name, expected in manifest["sha256"].items():
        if digest(snapshot / name) != expected:
            raise ValueError(f"PnP snapshot changed: {snapshot / name}")
    return manifest


def save(dataset):
    dataset = Path(dataset).resolve()
    snapshot = dataset / "training_pnp"
    sources = {name: dataset / relative for name, relative in FILES.items()}
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    source_hashes = {name: digest(path) for name, path in sources.items()}
    if snapshot.exists():
        manifest = verify(snapshot)
        if manifest["sha256"] != source_hashes:
            raise ValueError("Training PnP snapshot differs from current calibration or keypoints; "
                             "use a new session for a different box")
        return snapshot
    with tempfile.TemporaryDirectory(prefix=".training_pnp_", dir=dataset) as temporary:
        stage = Path(temporary) / "training_pnp"
        stage.mkdir()
        for name, source in sources.items():
            shutil.copy2(source, stage / name)
        if {name: digest(stage / name) for name in FILES} != source_hashes:
            raise ValueError("PnP inputs changed while saving the snapshot")
        calibration = json.loads((stage / "calibration.json").read_text())
        manifest = {
            "schema_version": 1,
            "calibration_revision": calibration["revision"],
            "sha256": source_hashes,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(snapshot)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("save", "verify"))
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        snapshot = args.dataset.resolve() / "training_pnp"
        if args.action == "save":
            snapshot = save(args.dataset)
        else:
            verify(snapshot)
        print(f"PnP snapshot {args.action} OK: {snapshot}")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"PnP snapshot {args.action} failed: {exc}\n")


if __name__ == "__main__":
    main()
