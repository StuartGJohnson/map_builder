from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from map_builder.mesh_processing import (
    MeshLike,
    load_and_transform_mesh,
    mesh_xy_bounds,
    rasterize_mesh_floor_free,
    rasterize_mesh_section,
    rasterize_mesh_slab,
)
from map_builder.occupancy_map import OccupancyMap, RasterGrid, write_ros_occupancy_map
from map_builder.pcd_io import read_point_cloud
from map_builder.pcd_preproc import PreprocessResult, preprocess_point_cloud


@dataclass(frozen=True)
class MeshMapContext:
    mesh: MeshLike
    grid: RasterGrid
    floor_map: OccupancyMap
    preprocessing_result: PreprocessResult


def prepare_mesh_map_context(
    mesh_path: str | Path,
    point_cloud_path: str | Path,
    *,
    ceiling_height: float,
    resolution: float,
    grid_padding: float = 0.5,
    z_max_floor: float = 0.05,
    floor_voxel_scale: float | None = None,
    min_up_normal_z: float = 0.5,
    preprocessing_options: dict | None = None,
) -> MeshMapContext:
    """Align mesh and point cloud, create a grid, and rasterize floor/free cells."""

    pcd = read_point_cloud(point_cloud_path)
    options = dict(preprocessing_options or {})
    options["target_ceiling_height"] = ceiling_height
    processed, preprocessing_result = preprocess_point_cloud(pcd, **options)
    del processed
    del pcd

    mesh = load_and_transform_mesh(mesh_path, preprocessing_result)
    grid = RasterGrid.from_bounds(
        mesh_xy_bounds(mesh),
        resolution=resolution,
        padding=grid_padding,
    )
    floor_map = rasterize_mesh_floor_free(
        mesh,
        z_max_floor=z_max_floor,
        grid=grid,
        voxel_scale=floor_voxel_scale or resolution,
        min_up_normal_z=min_up_normal_z,
    )
    return MeshMapContext(
        mesh=mesh,
        grid=grid,
        floor_map=floor_map,
        preprocessing_result=preprocessing_result,
    )


def generate_section_map(
    context: MeshMapContext,
    *,
    section_height: float,
    footprint_width: float | None = None,
) -> OccupancyMap:
    return rasterize_mesh_section(
        context.mesh,
        z_height=section_height,
        grid=context.grid,
        base_map=context.floor_map,
        footprint_width=footprint_width,
    )


def generate_slab_map(
    context: MeshMapContext,
    *,
    z_min: float,
    z_max: float,
    voxel_scale: float | None = None,
) -> OccupancyMap:
    return rasterize_mesh_slab(
        context.mesh,
        z_min=z_min,
        z_max=z_max,
        grid=context.grid,
        voxel_scale=voxel_scale or context.grid.resolution,
        base_map=context.floor_map,
    )


def write_occupancy_artifacts(
    output_base: str | Path,
    occupancy_map: OccupancyMap,
) -> tuple[Path, Path, Path]:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    npy_path = output_base.with_suffix(".npy")
    np.save(npy_path, occupancy_map.values)
    png_path, yaml_path = write_ros_occupancy_map(output_base, occupancy_map)
    return npy_path, png_path, yaml_path
