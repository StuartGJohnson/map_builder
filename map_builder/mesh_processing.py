from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import trimesh

from map_builder.occupancy_map import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyMap,
    RasterGrid,
    empty_occupancy_map,
)
from map_builder.pcd_preproc import PreprocessResult


MeshLike = trimesh.Trimesh | trimesh.Scene


def load_and_transform_mesh(
    input_path: str | Path,
    preprocessing_result: PreprocessResult,
    *,
    process: bool = False,
) -> MeshLike:
    """Load a mesh and apply the point-cloud source-to-processed transform."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    mesh = trimesh.load(input_path, force="scene", process=process)
    matrix = np.asarray(
        preprocessing_result.source_to_processed_transform,
        dtype=np.float64,
    )
    if matrix.shape != (4, 4):
        raise ValueError("preprocessing transform must have shape (4, 4)")
    mesh.apply_transform(matrix)
    return mesh


def mesh_xy_bounds(mesh: MeshLike) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return bounds[:, :2]


def rasterize_mesh_section(
    mesh: MeshLike,
    z_height: float,
    grid: RasterGrid,
    *,
    base_map: OccupancyMap | None = None,
    footprint_width: float | None = None,
) -> OccupancyMap:
    """Rasterize mesh edges intersected by a horizontal sampling plane."""

    result = _initialize_occupancy_map(grid, base_map)
    footprint_width = footprint_width or grid.resolution
    if footprint_width <= 0.0:
        raise ValueError("footprint_width must be positive")
    for geometry in _iter_geometry(mesh):
        section = geometry.section(
            plane_origin=np.array([0.0, 0.0, z_height]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        if section is None:
            continue

        planar, to_3d = section.to_2D()
        for entity in planar.entities:
            polyline = entity.discrete(planar.vertices)
            planar_3d = np.column_stack(
                (polyline, np.zeros(polyline.shape[0], dtype=np.float64))
            )
            world = trimesh.transform_points(planar_3d, to_3d)
            sampled = _sample_polyline_xy(world[:, :2], grid.resolution * 0.5)
            _rasterize_square_footprints(
                result,
                centers_xy=sampled,
                footprint_size=footprint_width,
            )
    return result


def rasterize_mesh_slab(
    mesh: MeshLike,
    z_min: float,
    z_max: float,
    grid: RasterGrid,
    *,
    voxel_scale: float,
    base_map: OccupancyMap | None = None,
) -> OccupancyMap:
    """Slice a mesh between two Z planes, voxelize it, and project across Z."""

    if z_max <= z_min:
        raise ValueError("z_max must be greater than z_min")
    if voxel_scale <= 0.0:
        raise ValueError("voxel_scale must be positive")

    result = _initialize_occupancy_map(grid, base_map)
    for geometry in _iter_geometry(mesh):
        slab = geometry.slice_plane(
            plane_origin=np.array([0.0, 0.0, z_min]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        if slab is None or slab.is_empty:
            continue
        slab = slab.slice_plane(
            plane_origin=np.array([0.0, 0.0, z_max]),
            plane_normal=np.array([0.0, 0.0, -1.0]),
        )
        if slab is None or slab.is_empty:
            continue

        voxels = slab.voxelized(voxel_scale)
        if voxels.filled_count == 0:
            continue
        occupied_xy_indices = np.argwhere(voxels.matrix.any(axis=2))
        voxel_indices = np.column_stack(
            (
                occupied_xy_indices,
                np.zeros(occupied_xy_indices.shape[0], dtype=int),
            )
        )
        occupied_xy = voxels.indices_to_points(voxel_indices)[:, :2]
        _rasterize_square_footprints(
            result,
            centers_xy=occupied_xy,
            footprint_size=voxel_scale,
        )
    return result


def rasterize_mesh_floor_free(
    mesh: MeshLike,
    z_max_floor: float,
    grid: RasterGrid,
    *,
    voxel_scale: float,
    min_up_normal_z: float = 0.5,
) -> OccupancyMap:
    """Project upward-facing mesh surfaces near z=0 as free map cells."""

    if z_max_floor <= 0.0:
        raise ValueError("z_max_floor must be positive")
    if voxel_scale <= 0.0:
        raise ValueError("voxel_scale must be positive")
    if not 0.0 < min_up_normal_z <= 1.0:
        raise ValueError("min_up_normal_z must be in the interval (0, 1]")

    result = empty_occupancy_map(grid, initial_value=UNKNOWN)
    for geometry in _iter_geometry(mesh):
        floor_band = _slice_z_band(geometry, -z_max_floor, z_max_floor)
        if floor_band is None:
            continue

        upward_faces = floor_band.face_normals[:, 2] >= min_up_normal_z
        if not np.any(upward_faces):
            continue
        upward_surface = trimesh.Trimesh(
            vertices=floor_band.vertices,
            faces=floor_band.faces[upward_faces],
            process=False,
        )
        voxels = upward_surface.voxelized(voxel_scale)
        if voxels.filled_count == 0:
            continue
        _project_voxels_xy(
            occupancy_map=result,
            voxels=voxels,
            footprint_size=voxel_scale,
            value=FREE,
        )
    return result


def _iter_geometry(mesh: MeshLike) -> Iterator[trimesh.Trimesh]:
    if isinstance(mesh, trimesh.Scene):
        for node_name in mesh.graph.nodes_geometry:
            transform, geometry_name = mesh.graph[node_name]
            geometry = mesh.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            yield geometry
    else:
        yield mesh


def _initialize_occupancy_map(
    grid: RasterGrid,
    base_map: OccupancyMap | None,
) -> OccupancyMap:
    if base_map is None:
        return empty_occupancy_map(grid, initial_value=UNKNOWN)
    if base_map.grid != grid:
        raise ValueError("base occupancy map grid does not match raster grid")
    return OccupancyMap(grid=grid, values=base_map.values.copy())


def _slice_z_band(
    geometry: trimesh.Trimesh,
    z_min: float,
    z_max: float,
) -> trimesh.Trimesh | None:
    band = geometry.slice_plane(
        plane_origin=np.array([0.0, 0.0, z_min]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    if band is None or band.is_empty:
        return None
    band = band.slice_plane(
        plane_origin=np.array([0.0, 0.0, z_max]),
        plane_normal=np.array([0.0, 0.0, -1.0]),
    )
    if band is None or band.is_empty:
        return None
    return band


def _project_voxels_xy(
    occupancy_map: OccupancyMap,
    voxels: trimesh.voxel.VoxelGrid,
    *,
    footprint_size: float,
    value: np.int8,
) -> None:
    occupied_xy_indices = np.argwhere(voxels.matrix.any(axis=2))
    voxel_indices = np.column_stack(
        (
            occupied_xy_indices,
            np.zeros(occupied_xy_indices.shape[0], dtype=int),
        )
    )
    occupied_xy = voxels.indices_to_points(voxel_indices)[:, :2]
    _rasterize_square_footprints(
        occupancy_map,
        centers_xy=occupied_xy,
        footprint_size=footprint_size,
        value=value,
    )


def _sample_polyline_xy(points: np.ndarray, spacing: float) -> np.ndarray:
    if points.shape[0] < 2:
        return points

    sampled = []
    for start, end in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(int(np.ceil(distance / spacing)), 1)
        weights = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
        sampled.append(start + (end - start) * weights)
    sampled.append(points[-1:])
    return np.vstack(sampled)


def _rasterize_square_footprints(
    occupancy_map: OccupancyMap,
    *,
    centers_xy: np.ndarray,
    footprint_size: float,
    value: np.int8 = OCCUPIED,
) -> None:
    """Mark every grid cell overlapped by square footprints centered at XY."""

    if centers_xy.size == 0:
        return

    grid = occupancy_map.grid
    half_size = footprint_size * 0.5
    epsilon = np.finfo(np.float64).eps * max(
        1.0,
        abs(grid.origin_x),
        abs(grid.origin_y),
        footprint_size,
    )

    min_cols = np.floor(
        (centers_xy[:, 0] - half_size - grid.origin_x) / grid.resolution
    ).astype(int)
    max_cols = np.floor(
        (centers_xy[:, 0] + half_size - epsilon - grid.origin_x) / grid.resolution
    ).astype(int)
    min_rows = np.floor(
        (centers_xy[:, 1] - half_size - grid.origin_y) / grid.resolution
    ).astype(int)
    max_rows = np.floor(
        (centers_xy[:, 1] + half_size - epsilon - grid.origin_y) / grid.resolution
    ).astype(int)

    overlaps_grid = (
        (max_cols >= 0)
        & (min_cols < grid.width)
        & (max_rows >= 0)
        & (min_rows < grid.height)
    )
    min_cols = min_cols[overlaps_grid]
    max_cols = max_cols[overlaps_grid]
    min_rows = min_rows[overlaps_grid]
    max_rows = max_rows[overlaps_grid]

    min_cols = np.clip(min_cols, 0, grid.width - 1)
    max_cols = np.clip(max_cols, 0, grid.width - 1)
    min_rows = np.clip(min_rows, 0, grid.height - 1)
    max_rows = np.clip(max_rows, 0, grid.height - 1)

    for row_min, row_max, col_min, col_max in zip(
        min_rows,
        max_rows,
        min_cols,
        max_cols,
    ):
        if row_min <= row_max and col_min <= col_max:
            occupancy_map.values[
                row_min : row_max + 1,
                col_min : col_max + 1,
            ] = value
