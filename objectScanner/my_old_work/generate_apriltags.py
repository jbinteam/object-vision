#!/usr/bin/env python3
"""Generate AprilTag images for every family (or a chosen subset).

By default it writes one sample tag (id 0) per family plus an overview
montage of all families. Use --count / --ids / --all to generate more.

Tags are upscaled with nearest-neighbour (crisp squares) and given a white
"quiet zone" border, which detectors need when the tag is printed.

There are 9 families across two AprilTag generations:
    v2: tag16h5, tag25h9, tag36h10, tag36h11
    v3: tagCircle21h7, tagCircle49h12, tagCustom48h12,
        tagStandard41h12, tagStandard52h13

Examples:
    python generate_apriltags.py                         # one of each + montage
    python generate_apriltags.py --count 5               # ids 0..4 of each family
    python generate_apriltags.py --families tag36h11 --ids 0,1,7,42
    python generate_apriltags.py --families tag36h11 --all   # every id (587!)

Requires:
    pip install moms-apriltag opencv-python numpy
"""

import argparse
import os
import sys

import cv2
import numpy as np

try:
    import moms_apriltag as mat
except ImportError:
    sys.exit(
        "Missing dependency. Install it with:\n"
        "    pip install moms-apriltag opencv-python numpy"
    )

# Map each family name to the generator generation that produces it.
FAMILIES = {name: 2 for name in mat.apriltags_v2}
FAMILIES.update({name: 3 for name in mat.apriltags_v3})


def make_generator(family: str):
    """Return a TagGenerator for the given family (picks v2 vs v3)."""
    gen = mat.TagGenerator2 if FAMILIES[family] == 2 else mat.TagGenerator3
    return gen(family)


def render_tag(raw: np.ndarray, px: int, quiet_cells: int,
               transparent_white: bool = True) -> np.ndarray:
    """Upscale a raw tag bitmap to ~px pixels and add a white quiet zone.

    The quiet zone is sized as a multiple of the tag's cell size so it scales
    with the tag, matching AprilTag printing guidelines (>= 1 cell border).

    Circle/custom families mark cells outside the tag shape with a grey
    "transparent" value (127); for a printable tag these become white.
    """
    raw = raw.astype(np.uint8)
    if transparent_white:
        raw = np.where((raw != 0) & (raw != 255), 255, raw).astype(np.uint8)
    cells = raw.shape[0]
    scale = max(1, px // cells)
    big = cv2.resize(raw, (cells * scale, cells * scale),
                     interpolation=cv2.INTER_NEAREST)
    border = quiet_cells * scale
    return cv2.copyMakeBorder(
        big, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255,
    )


def parse_ids(args, max_id: int):
    """Resolve which tag ids to generate for a family, given CLI args."""
    if args.all:
        return range(max_id)
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip() != ""]
        bad = [i for i in ids if i < 0 or i >= max_id]
        if bad:
            print(f"    skipping out-of-range ids {bad} (max {max_id - 1})")
        return [i for i in ids if 0 <= i < max_id]
    return range(min(args.count, max_id))


def build_montage(samples, cols: int) -> np.ndarray:
    """Lay out (label, image) samples into a labelled grid contact sheet."""
    cell = max(img.shape[0] for _, img in samples)
    pad, label_h = 20, 30
    tile = cell + 2 * pad
    rows = (len(samples) + cols - 1) // cols
    sheet = np.full((rows * (tile + label_h), cols * tile), 255, np.uint8)

    for idx, (label, img) in enumerate(samples):
        r, c = divmod(idx, cols)
        # Center the (possibly smaller) tag within its tile.
        img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_NEAREST)
        y0 = r * (tile + label_h) + pad
        x0 = c * tile + pad
        sheet[y0:y0 + cell, x0:x0 + cell] = img
        cv2.putText(
            sheet, label, (c * tile + 8, y0 + cell + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA,
        )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AprilTag images.")
    parser.add_argument(
        "--families", default="all",
        help="Comma-separated family names, or 'all' (default).",
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--count", type=int, default=1,
                     help="Generate ids 0..count-1 per family (default: 1).")
    grp.add_argument("--ids", help="Comma-separated specific ids, e.g. 0,1,7,42.")
    grp.add_argument("--all", action="store_true",
                     help="Generate every id in each family (can be huge).")
    parser.add_argument("--size", type=int, default=600,
                        help="Approx tag size in pixels before border (default: 600).")
    parser.add_argument("--quiet-cells", type=int, default=1,
                        help="White border width in tag cells (default: 1).")
    parser.add_argument("--out", default="apriltags",
                        help="Output directory (default: ./apriltags).")
    parser.add_argument("--keep-transparent", action="store_true",
                        help="Keep grey transparent cells instead of making them white.")
    parser.add_argument("--no-montage", action="store_true",
                        help="Skip the overview montage.")
    args = parser.parse_args()

    if args.families == "all":
        families = list(FAMILIES)
    else:
        families = [f.strip() for f in args.families.split(",")]
        bad = [f for f in families if f not in FAMILIES]
        if bad:
            sys.exit(f"Unknown families: {bad}\nAvailable: {list(FAMILIES)}")

    os.makedirs(args.out, exist_ok=True)
    montage_samples, total = [], 0

    for family in families:
        tg = make_generator(family)
        fam_dir = os.path.join(args.out, family)
        os.makedirs(fam_dir, exist_ok=True)
        ids = list(parse_ids(args, tg.max_id))

        print(f"{family}: {tg.max_id} tags available -> generating {len(ids)}")

        first_img = None
        for tag_id in ids:
            raw = tg.generate(tag_id)
            img = render_tag(raw, args.size, args.quiet_cells,
                             transparent_white=not args.keep_transparent)
            path = os.path.join(fam_dir, f"{family}_{tag_id:05d}.png")
            cv2.imwrite(path, img)
            total += 1
            if first_img is None:
                first_img = img

        if first_img is not None:
            montage_samples.append((f"{family} (n={tg.max_id})", first_img))

    print(f"\nWrote {total} tag image(s) under {args.out}/")

    if not args.no_montage and montage_samples:
        cols = min(3, len(montage_samples))
        sheet = build_montage(montage_samples, cols)
        mpath = os.path.join(args.out, "_families_overview.png")
        cv2.imwrite(mpath, sheet)
        print(f"Wrote family overview montage to {mpath}")


if __name__ == "__main__":
    main()
