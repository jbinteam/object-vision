"""Frame-to-model TSDF reconstruction, adapted from the Orbbec reference.

See NOTICE and LICENSE. All reconstruction coordinates are in metres.
"""
import numpy as np
import logging
import open3d as o3d
import open3d.core as o3c

logger = logging.getLogger("object_scanner")


def plausible_motion(delta):
    """Reject invalid rigid transforms and large single-frame tracking jumps."""
    if delta.shape != (4, 4) or not np.isfinite(delta).all():
        return False
    rotation = delta[:3, :3]
    return (np.allclose(delta[3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            and abs(np.linalg.det(rotation) - 1) < 1e-3
            and np.linalg.norm(delta[:3, 3]) <= 0.25
            and np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)) <= np.pi / 4)

class TensorSlam:
    """Open3D tensor dense SLAM: GPU voxel-hashing map + frame-to-model tracking.

    The map is a spatially-hashed TSDF (VoxelBlockGrid wrapped by slam.Model).
    Each frame is tracked by ray-casting the model to a synthetic depth/colour
    frame at the last pose and aligning the new frame to it (multi-scale tensor
    RGB-D odometry). `pose` is T_world_cam (camera -> world) for the current
    frame, in METRES; the first frame defines the world origin.
    """

    # Per-pyramid-level iteration counts for the multi-scale odometry
    # (coarse -> fine). The Open3D defaults; coarse levels do most of the work.
    _CRITERIA = [
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(6),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(3),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(1),
    ]

    def __init__(self, intrinsic_t, width, height, voxel_size_mm, depth_trunc_m,
                 device, block_count, min_fitness):
        self.device = device
        self.depth_scale = 1000.0                 # our depth images are in mm
        self.depth_max = depth_trunc_m            # metres
        self.depth_min = 0.1                       # metres (ray-cast near clip)
        self.trunc_vx = 8.0                        # TSDF truncation, in voxels
        self.min_fitness = min_fitness

        self.model = o3d.t.pipelines.slam.Model(
            voxel_size_mm / 1000.0,                # metres
            16,                                    # 16^3 voxels per block
            block_count,
            o3c.Tensor(np.identity(4)),
            device,
        )
        self.input_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic_t, device)
        self.raycast_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic_t, device)

        self.pose = np.identity(4)                 # T_world_cam (metres)
        self.frame_idx = 0
        self.trajectory = []                       # list of 4x4 T_world_cam (m)
        self.tracked_frames = 0
        self.lost_frames = 0
        self.last_fitness = 1.0

    def process(self, color_rgb, depth_mm):
        """Upload one RGB-D frame, track it against the model, then integrate.

        `color_rgb` is HxWx3 uint8 (RGB), `depth_mm` is HxW uint16 (millimetres).
        Returns True if the frame was tracked & mapped, False if tracking was
        lost (the frame is skipped so a bad pose can't corrupt the map).
        """
        # Do not seed an empty volume: it cannot provide a tracking reference.
        valid = (depth_mm > 0) & (depth_mm <= self.depth_max * 1000)
        if np.count_nonzero(valid) < 100:
            self.last_fitness = 0.0
            self.lost_frames += 1
            return False
        logger.info("Upload RGB-D frame")
        color_img = o3d.t.geometry.Image(o3c.Tensor(color_rgb, device=self.device))
        depth_img = o3d.t.geometry.Image(o3c.Tensor(depth_mm, device=self.device))
        self.input_frame.set_data_from_image("color", color_img)
        self.input_frame.set_data_from_image("depth", depth_img)

        if self.frame_idx > 0:
            # FRAME-TO-MODEL: align the new frame to the ray-cast model frame.
            # Open3D RAISES (singular 6x6 system) when a frame has too little
            # geometric structure to track — e.g. nothing within depth range or
            # a featureless surface — so treat that as a lost frame, not a crash.
            try:
                logger.info("Track frame against model")
                result = self.model.track_frame_to_model(
                    self.input_frame,
                    self.raycast_frame,
                    self.depth_scale,
                    self.depth_max,
                    method=o3d.t.pipelines.odometry.Method.PointToPlane,
                    criteria=self._CRITERIA,
                )
            except RuntimeError as exc:
                # Only a known geometric tracking failure is recoverable.
                # Never retry CUDA faults as if they were lost tracking.
                if "singular 6x6 linear system" not in str(exc).lower():
                    raise
                self.last_fitness = 0.0
                self.lost_frames += 1
                return False
            self.last_fitness = float(result.fitness)
            if not np.isfinite(self.last_fitness) or self.last_fitness < self.min_fitness:
                # Lost: keep the previous model frame as the reference and skip
                # this frame rather than fusing it at a wrong pose.
                self.lost_frames += 1
                return False
            delta = result.transformation.cpu().numpy()
            if not plausible_motion(delta):
                logger.warning("Rejected invalid pose or motion exceeding 25 cm / 45 degrees per frame")
                self.lost_frames += 1
                return False
            self.pose = self.pose @ delta

        # Commit: record the pose, fuse the frame, and re-synthesize the model
        # frame for the next track step.
        logger.info("Update model pose")
        self.model.update_frame_pose(self.frame_idx, o3c.Tensor(self.pose))
        logger.info("Integrate depth into TSDF")
        self.model.integrate(self.input_frame, self.depth_scale, self.depth_max, self.trunc_vx)
        logger.info("Raycast fused model")
        self.model.synthesize_model_frame(
            self.raycast_frame, self.depth_scale, self.depth_min, self.depth_max,
            self.trunc_vx, False,
        )

        self.trajectory.append(np.copy(self.pose))
        self.frame_idx += 1
        self.tracked_frames += 1
        logger.info("Frame complete")
        return True

    def extract_point_cloud(self):
        """Fused point cloud as a legacy PointCloud, in metres (SLAM units)."""
        return self.model.extract_pointcloud().to_legacy()

    def extract_mesh(self):
        """Fused triangle mesh as a legacy TriangleMesh, in metres (SLAM units)."""
        mesh = self.model.extract_trianglemesh().to_legacy()
        mesh.compute_vertex_normals()
        return mesh


def _isolate_object_cloud(pcd, remove_plane, plane_thresh_m, cluster_eps_m,
                          min_points):
    """Return (object_cloud, info_str): drop the support plane, keep the object.

    1. If `remove_plane`, fit the dominant plane with RANSAC and remove it — but
       only when it accounts for a substantial share of the points (so we don't
       accidentally delete a flat-topped object).
    2. Cluster what remains (DBSCAN) and keep the largest cluster as the object.
    All distances are in metres (the cloud's units).
    """
    info = []
    n0 = len(pcd.points)
    if n0 < 100:
        return pcd, "too few points to isolate"

    if remove_plane:
        plane, inliers = pcd.segment_plane(
            distance_threshold=plane_thresh_m, ransac_n=3, num_iterations=1000
        )
        # Only strip the plane if it's a big chunk of the scene (a real table /
        # floor), not a small facet of the object itself.
        if 0.15 * n0 <= len(inliers) <= 0.95 * n0:
            pcd = pcd.select_by_index(inliers, invert=True)
            info.append(f"removed plane ({len(inliers)} pts)")
        else:
            info.append("no dominant plane removed")

    if len(pcd.points) > min_points:
        labels = np.asarray(
            pcd.cluster_dbscan(eps=cluster_eps_m, min_points=min_points)
        )
        if labels.max() >= 0:
            counts = np.bincount(labels[labels >= 0])
            biggest = int(np.argmax(counts))
            keep = np.where(labels == biggest)[0]
            pcd = pcd.select_by_index(keep)
            info.append(f"largest cluster {len(keep)} pts ({labels.max() + 1} found)")

    return pcd, "; ".join(info) if info else "no isolation applied"


def _crop_mesh_to_object(mesh, object_cloud, margin_m):
    """Crop `mesh` to the object's bounding box, then keep its largest component."""
    if len(object_cloud.points) == 0 or len(mesh.vertices) == 0:
        return mesh

    bbox = object_cloud.get_axis_aligned_bounding_box()
    bbox.min_bound = np.asarray(bbox.min_bound) - margin_m
    bbox.max_bound = np.asarray(bbox.max_bound) + margin_m
    mesh = mesh.crop(bbox)
    if len(mesh.triangles) == 0:
        return mesh

    # Keep only the largest connected surface (drops floating speckles).
    tri_clusters, n_tri, _areas = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    n_tri = np.asarray(n_tri)
    if len(n_tri) > 1:
        largest = int(np.argmax(n_tri))
        mesh.remove_triangles_by_mask(tri_clusters != largest)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh

