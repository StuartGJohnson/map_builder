from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import open3d as o3d

from map_builder.pcd_io import read_point_cloud, write_point_cloud


@dataclass(frozen=True)
class FilterResult:
    """Point counts after each filtering operation."""

    input_point_count: int
    voxel_point_count: int
    statistical_point_count: int
    radius_point_count: int


def filter_point_cloud(
    pcd: o3d.t.geometry.PointCloud,
    *,
    voxel_size: float = 0.01,
    statistical_neighbors: int = 30,
    statistical_std_ratio: float = 2.0,
    radius_nb_points: int = 6,
    radius_search_radius: float = 0.04,
) -> tuple[o3d.t.geometry.PointCloud, FilterResult]:
    """Voxelize a point cloud and remove statistical and radius outliers."""

    input_count = _point_count(pcd)
    if input_count == 0:
        raise ValueError("input point cloud is empty")

    filtered = pcd.voxel_down_sample(voxel_size)
    voxel_count = _point_count(filtered)

    filtered, _ = filtered.remove_statistical_outliers(
        nb_neighbors=statistical_neighbors,
        std_ratio=statistical_std_ratio,
    )
    statistical_count = _point_count(filtered)

    filtered, _ = filtered.remove_radius_outliers(
        nb_points=radius_nb_points,
        search_radius=radius_search_radius,
    )
    radius_count = _point_count(filtered)

    return filtered, FilterResult(
        input_point_count=input_count,
        voxel_point_count=voxel_count,
        statistical_point_count=statistical_count,
        radius_point_count=radius_count,
    )


def filter_point_cloud_file(
    input_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> FilterResult:
    """Read, filter, and write a point cloud."""

    pcd = read_point_cloud(input_path)
    filtered, result = filter_point_cloud(pcd, **kwargs)
    write_point_cloud(output_path, filtered)
    return result


def _point_count(pcd: o3d.t.geometry.PointCloud) -> int:
    return int(pcd.point.positions.shape[0])
