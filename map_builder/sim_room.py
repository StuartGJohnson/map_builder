from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def create_sim_room_mesh(
    *,
    half_width: float = 2.0,
    ceiling_height: float = 2.0,
) -> trimesh.Trimesh:
    """Create a closed rectangular room mesh with inward-facing normals."""

    mesh = trimesh.creation.box(
        extents=[2.0 * half_width, 2.0 * half_width, ceiling_height],
        transform=trimesh.transformations.translation_matrix(
            [0.0, 0.0, ceiling_height * 0.5]
        ),
    )
    mesh.invert()
    return mesh


def create_sim_room_point_cloud(
    *,
    half_width: float = 2.0,
    ceiling_height: float = 2.0,
    point_interval: float = 0.1,
) -> o3d.t.geometry.PointCloud:
    """Create a deterministic room-surface point cloud with inward normals."""

    if point_interval <= 0.0:
        raise ValueError("point_interval must be positive")

    xy = _inclusive_axis(-half_width, half_width, point_interval)
    z = _inclusive_axis(0.0, ceiling_height, point_interval)
    points = []
    normals = []

    _append_plane(points, normals, xy, xy, 0.0, axis=2, normal=[0.0, 0.0, 1.0])

    # Slightly coarser non-floor surfaces make the floor the unambiguous
    # largest plane for preprocessing RANSAC.
    coarse = point_interval * 1.2
    xy_coarse = _inclusive_axis(-half_width, half_width, coarse)
    z_coarse = _inclusive_axis(0.0, ceiling_height, coarse)
    _append_plane(
        points,
        normals,
        xy_coarse,
        xy_coarse,
        ceiling_height,
        axis=2,
        normal=[0.0, 0.0, -1.0],
    )
    _append_plane(
        points,
        normals,
        xy_coarse,
        z_coarse,
        -half_width,
        axis=0,
        normal=[1.0, 0.0, 0.0],
    )
    _append_plane(
        points,
        normals,
        xy_coarse,
        z_coarse,
        half_width,
        axis=0,
        normal=[-1.0, 0.0, 0.0],
    )
    _append_plane(
        points,
        normals,
        xy_coarse,
        z_coarse,
        -half_width,
        axis=1,
        normal=[0.0, 1.0, 0.0],
    )
    _append_plane(
        points,
        normals,
        xy_coarse,
        z_coarse,
        half_width,
        axis=1,
        normal=[0.0, -1.0, 0.0],
    )

    positions = np.vstack(points).astype(np.float32)
    point_normals = np.vstack(normals).astype(np.float32)
    colors = np.full((positions.shape[0], 3), 180, dtype=np.uint8)
    classes = np.zeros((positions.shape[0], 1), dtype=np.uint8)
    pcd = o3d.t.geometry.PointCloud(
        {
            "positions": o3d.core.Tensor(positions),
            "normals": o3d.core.Tensor(point_normals),
            "colors": o3d.core.Tensor(colors),
        }
    )
    pcd.point["class"] = o3d.core.Tensor(classes)
    return pcd


def write_sim_room_fixtures(
    output_dir: str | Path,
    *,
    half_width: float = 2.0,
    ceiling_height: float = 2.0,
    point_interval: float = 0.1,
) -> tuple[Path, Path]:
    """Write deterministic sim-room OBJ and PLY fixtures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = output_dir / "sim_room.obj"
    ply_path = output_dir / "sim_room.ply"

    create_sim_room_mesh(
        half_width=half_width,
        ceiling_height=ceiling_height,
    ).export(obj_path)
    pcd = create_sim_room_point_cloud(
        half_width=half_width,
        ceiling_height=ceiling_height,
        point_interval=point_interval,
    )
    if not o3d.t.io.write_point_cloud(str(ply_path), pcd):
        raise RuntimeError(f"failed to write point cloud to {ply_path}")
    return obj_path, ply_path


def _inclusive_axis(lower: float, upper: float, spacing: float) -> np.ndarray:
    count = int(np.ceil((upper - lower) / spacing)) + 1
    return np.linspace(lower, upper, count)


def _append_plane(
    points: list[np.ndarray],
    normals: list[np.ndarray],
    first_axis: np.ndarray,
    second_axis: np.ndarray,
    fixed: float,
    *,
    axis: int,
    normal: list[float],
) -> None:
    first, second = np.meshgrid(first_axis, second_axis, indexing="xy")
    plane = np.empty((first.size, 3), dtype=np.float64)
    variable_axes = [index for index in range(3) if index != axis]
    plane[:, axis] = fixed
    plane[:, variable_axes[0]] = first.ravel()
    plane[:, variable_axes[1]] = second.ravel()
    points.append(plane)
    normals.append(np.tile(np.asarray(normal), (plane.shape[0], 1)))
