#!/usr/bin/env python3
"""Interactive colored point-cloud viewer and live scanner preview."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import open3d as o3d


class CloudPreview:
    def __init__(self, title):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window(title, width=960, height=720):
            raise RuntimeError("Cannot open 3D window. Use --no-view or --no-3d.")
        self.cloud = o3d.geometry.PointCloud()
        self.added = False
        self.running = True
        self.original_colors = None
        self.grey = False
        self.vis.get_render_option().background_color = [0.055, 0.065, 0.085]
        self.vis.get_render_option().point_size = 2
        self.vis.register_key_callback(ord("Q"), self.quit)
        self.vis.register_key_callback(256, self.quit)
        self.vis.register_key_callback(ord("C"), self.toggle_color)
        self.vis.register_key_callback(ord("R"), self.reset)

    def quit(self, _):
        self.running = False
        return False

    def reset(self, _):
        self.vis.reset_view_point(True)
        self.orient()
        return False

    def orient(self):
        view = self.vis.get_view_control()
        view.set_front([0, 0, -1])
        view.set_up([0, -1, 0])
        view.set_lookat(self.cloud.get_center())

    def toggle_color(self, _):
        self.grey = not self.grey
        self.apply_colors()
        return False

    def apply_colors(self):
        if self.grey or self.original_colors is None:
            self.cloud.paint_uniform_color([0.7, 0.7, 0.7])
        else:
            self.cloud.colors = o3d.utility.Vector3dVector(self.original_colors)
        self.vis.update_geometry(self.cloud)

    def update(self, cloud):
        if not len(cloud.points):
            return
        self.cloud.points = cloud.points
        self.original_colors = np.asarray(cloud.colors).copy() if cloud.has_colors() else None
        if not self.added:
            self.vis.add_geometry(self.cloud)
            self.added = True
            self.orient()
        self.apply_colors()

    def poll(self):
        if not self.running:
            return False
        # poll_events invokes key callbacks synchronously. Read self.running
        # afterward so Q/Esc cannot be overwritten by a True poll result.
        window_open = self.vis.poll_events()
        self.running = self.running and window_open
        if self.running:
            self.vis.update_renderer()
        return self.running

    def close(self):
        self.running = False
        self.vis.destroy_window()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--voxel", type=float, default=0, help="Display downsampling in mm")
    args = p.parse_args()
    if not args.path.is_file():
        p.error(f"File not found: {args.path}")
    cloud = o3d.io.read_point_cloud(str(args.path))
    if not len(cloud.points):
        p.error("File contains no readable points")
    if args.voxel > 0:
        cloud = cloud.voxel_down_sample(args.voxel / 1000)
    print(f"{len(cloud.points):,} points; extent (m): {cloud.get_axis_aligned_bounding_box().get_extent()}")
    print("Mouse: rotate/pan/zoom | C: color/grey | R: reset | +/-: point size | Q: quit")
    preview = CloudPreview(args.path.name)
    try:
        preview.update(cloud)
        import time
        while preview.poll():
            time.sleep(0.01)
    finally:
        preview.close()


if __name__ == "__main__":
    main()
