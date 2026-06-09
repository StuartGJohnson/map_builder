from __future__ import annotations

from pathlib import Path

import open3d as o3d


def read_point_cloud(path: str | Path) -> o3d.t.geometry.PointCloud:
    """Read a point cloud with Open3D tensor I/O to preserve point attributes."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    pcd = o3d.t.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"point cloud is empty: {path}")
    return pcd


def write_point_cloud(path: str | Path, pcd: o3d.t.geometry.PointCloud) -> Path:
    """Write a point cloud and return the output path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.t.io.write_point_cloud(str(path), pcd):
        raise RuntimeError(f"failed to write point cloud to {path}")
    return path
