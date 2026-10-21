#!/usr/bin/env python3
"""Collect NOCS-ready RGB-D views with metric box poses from an RGB-D camera.

This entry point records rectified RGB, aligned uint16 millimetre depth, camera
intrinsics, a right-handed box-centred object pose, and metric box dimensions.
Instance masks and dense XYZ NOCS maps are generated afterward so foreground
segmentation can be reviewed before it becomes training supervision.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_turntable_dataset import arguments as _capture_arguments
from capture_turntable_dataset import main as _capture_main


def arguments(argv=None):
    return _capture_arguments(argv, description=__doc__)


def main(argv=None):
    return _capture_main(
        argv,
        description=__doc__,
        workflow_name="NOCS-ready RGB-D capture",
    )


if __name__ == "__main__":
    main()
