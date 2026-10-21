#!/usr/bin/env python3
"""Export a recorded objectScanner session to NVlabs CenterPose training pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from centerpose_dataset import export_centerpose_dataset, validate_category  # noqa: E402


def category_arg(value):
    try:
        return validate_category(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Recorded objectScanner dataset directory")
    parser.add_argument("--category", type=category_arg,
                        help="CenterPose category (default: dataset directory name)")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()
    category = args.category or category_arg(args.dataset.name)
    count, destination = export_centerpose_dataset(args.dataset, category, args.split)
    print(f"Exported {count} CenterPose samples to {destination}")


if __name__ == "__main__":
    main()
