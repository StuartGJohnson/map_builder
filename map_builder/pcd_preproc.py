from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

from map_builder.pcd_io import read_point_cloud, write_point_cloud


@dataclass(frozen=True)
class PreprocessResult:
    """Summary of the fitted geometry and transformed point cloud."""

    floor_plane: tuple[float, float, float, float]
    ceiling_height_before_scale: float
    scale: float
    source_to_processed_transform: tuple[tuple[float, float, float, float], ...]
    point_count: int


def preprocess_point_cloud(
    pcd: o3d.t.geometry.PointCloud,
    *,
    target_ceiling_height: float,
    floor_distance_threshold: float = 0.025,
    ceiling_distance_threshold: float = 0.025,
    ransac_n: int = 3,
    floor_iterations: int = 2000,
    ceiling_iterations: int = 2000,
    max_ransac_points: int = 300_000,
    random_seed: int = 7,
    chunk_size: int = 1_000_000,
) -> tuple[o3d.t.geometry.PointCloud, PreprocessResult]:
    """Align and scale a point cloud using floor and ceiling planes.

    Point positions are transformed so the detected floor is at z=0 and +z
    points to the side of the floor containing the most points. Normals, when
    present, are rotated into the same frame. The detected ceiling is scaled to
    ``target_ceiling_height`` meters.
    """

    positions = pcd.point.positions.numpy()
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("input point cloud must have Nx3 positions")
    if positions.shape[0] < ransac_n:
        raise ValueError("input point cloud does not contain enough points")
    if target_ceiling_height <= 0.0:
        raise ValueError("target_ceiling_height must be positive")

    rng = np.random.default_rng(random_seed)
    o3d.utility.random.seed(random_seed)
    sample_idx = _sample_indices(positions.shape[0], max_ransac_points, rng)
    sample_positions = np.asarray(positions[sample_idx], dtype=np.float64)

    floor_plane = _fit_floor_plane(
        sample_positions,
        distance_threshold=floor_distance_threshold,
        ransac_n=ransac_n,
        iterations=floor_iterations,
    )
    floor_plane = _orient_plane_to_most_points(floor_plane, positions, chunk_size)

    origin, basis = _floor_transform(floor_plane)
    transformed = _transform_positions(positions, origin, basis, chunk_size)

    _rotate_normals_if_present(pcd, basis)

    z_sample = transformed[sample_idx, 2].astype(np.float64, copy=False)
    ceiling_height = _fit_parallel_ceiling_height(
        z_sample,
        distance_threshold=ceiling_distance_threshold,
        iterations=ceiling_iterations,
        rng=rng,
    )
    if ceiling_height <= 0.0:
        raise ValueError(f"invalid ceiling height before scale: {ceiling_height}")

    scale = target_ceiling_height / ceiling_height
    _scale_positions_in_place(transformed, scale, chunk_size)
    source_to_processed_transform = _source_to_processed_transform(
        origin,
        basis,
        scale,
    )
    pcd.point.positions = o3d.core.Tensor(
        transformed,
        dtype=pcd.point.positions.dtype,
        device=pcd.point.positions.device,
    )

    return pcd, PreprocessResult(
        floor_plane=tuple(float(v) for v in floor_plane),
        ceiling_height_before_scale=float(ceiling_height),
        scale=float(scale),
        source_to_processed_transform=tuple(
            tuple(float(value) for value in row)
            for row in source_to_processed_transform
        ),
        point_count=int(pcd.point.positions.shape[0]),
    )


def preprocess_point_cloud_file(
    input_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> PreprocessResult:
    """Read, preprocess, and write a point cloud."""

    pcd = read_point_cloud(input_path)
    processed, result = preprocess_point_cloud(pcd, **kwargs)
    write_point_cloud(output_path, processed)
    return result


def _sample_indices(count: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= max_count:
        return np.arange(count)
    return rng.choice(count, size=max_count, replace=False)


def _fit_floor_plane(
    points: np.ndarray,
    *,
    distance_threshold: float,
    ransac_n: int,
    iterations: int,
) -> np.ndarray:
    legacy = o3d.geometry.PointCloud()
    legacy.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = legacy.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=iterations,
    )
    if len(inliers) == 0:
        raise ValueError("RANSAC did not find a floor plane")

    plane = np.asarray(plane_model, dtype=np.float64)
    norm = np.linalg.norm(plane[:3])
    if norm == 0.0:
        raise ValueError("RANSAC returned an invalid floor plane")
    return plane / norm


def _orient_plane_to_most_points(
    plane: np.ndarray,
    positions: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    positive = 0
    negative = 0
    normal = plane[:3]
    offset = plane[3]
    for chunk in _chunks(positions, chunk_size):
        signed = chunk.astype(np.float64, copy=False) @ normal + offset
        positive += int(np.count_nonzero(signed > 0.0))
        negative += int(np.count_nonzero(signed < 0.0))
    return plane if positive >= negative else -plane


def _floor_transform(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = plane[:3] / np.linalg.norm(plane[:3])
    origin = -plane[3] * z_axis

    candidate = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(candidate, z_axis))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])

    x_axis = candidate - np.dot(candidate, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return origin, np.column_stack((x_axis, y_axis, z_axis))


def _transform_positions(
    positions: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    transformed = np.empty_like(positions)
    for start in range(0, positions.shape[0], chunk_size):
        stop = min(start + chunk_size, positions.shape[0])
        transformed[start:stop] = (
            (positions[start:stop].astype(np.float64, copy=False) - origin) @ basis
        ).astype(positions.dtype, copy=False)
    return transformed


def _source_to_processed_transform(
    origin: np.ndarray,
    basis: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Return the 4x4 transform matching the point-coordinate operation."""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * basis.T
    transform[:3, 3] = -scale * basis.T @ origin
    return transform


def _rotate_normals_if_present(
    pcd: o3d.t.geometry.PointCloud,
    basis: np.ndarray,
) -> None:
    if "normals" not in pcd.point:
        return
    normals = pcd.point.normals.numpy()
    rotated = (normals.astype(np.float64, copy=False) @ basis).astype(
        normals.dtype, copy=False
    )
    pcd.point.normals = o3d.core.Tensor(
        rotated,
        dtype=pcd.point.normals.dtype,
        device=pcd.point.normals.device,
    )


def _fit_parallel_ceiling_height(
    z_values: np.ndarray,
    *,
    distance_threshold: float,
    iterations: int,
    rng: np.random.Generator,
    floor_exclusion: float = 0.20,
) -> float:
    candidates = z_values[np.isfinite(z_values) & (z_values > floor_exclusion)]
    if candidates.size == 0:
        raise ValueError("no points above the floor were available for ceiling fit")

    best_center = None
    best_count = -1
    for center in rng.choice(candidates, size=min(iterations, candidates.size), replace=False):
        count = int(np.count_nonzero(np.abs(candidates - center) <= distance_threshold))
        if count > best_count:
            best_center = float(center)
            best_count = count

    if best_center is None:
        raise ValueError("RANSAC did not find a ceiling plane")

    inliers = candidates[np.abs(candidates - best_center) <= distance_threshold]
    if inliers.size == 0:
        raise ValueError("RANSAC ceiling fit had no inliers")
    return float(np.median(inliers))


def _scale_positions_in_place(
    positions: np.ndarray,
    scale: float,
    chunk_size: int,
) -> None:
    for chunk in _chunks(positions, chunk_size):
        chunk *= scale


def _chunks(array: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, array.shape[0], chunk_size):
        stop = min(start + chunk_size, array.shape[0])
        yield array[start:stop]
