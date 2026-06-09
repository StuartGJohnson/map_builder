from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(100)


@dataclass(frozen=True)
class RasterGrid:
    """A fixed 2D grid in world XY coordinates."""

    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int

    @classmethod
    def from_bounds(
        cls,
        bounds_xy: np.ndarray,
        resolution: float,
        *,
        padding: float = 0.0,
    ) -> RasterGrid:
        bounds = np.asarray(bounds_xy, dtype=np.float64)
        if bounds.shape != (2, 2):
            raise ValueError("bounds_xy must have shape (2, 2)")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")

        lower = np.floor((bounds[0] - padding) / resolution) * resolution
        upper = np.ceil((bounds[1] + padding) / resolution) * resolution
        size = np.maximum(np.ceil((upper - lower) / resolution).astype(int), 1)
        return cls(
            resolution=float(resolution),
            origin_x=float(lower[0]),
            origin_y=float(lower[1]),
            width=int(size[0]),
            height=int(size[1]),
        )

    def cell_indices(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xy, dtype=np.float64)
        cols = np.floor((points[:, 0] - self.origin_x) / self.resolution).astype(int)
        rows = np.floor((points[:, 1] - self.origin_y) / self.resolution).astype(int)
        valid = (
            (cols >= 0)
            & (cols < self.width)
            & (rows >= 0)
            & (rows < self.height)
        )
        return rows[valid], cols[valid]


@dataclass(frozen=True)
class OccupancyMap:
    """ROS-style occupancy values on a fixed world-coordinate grid."""

    grid: RasterGrid
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (self.grid.height, self.grid.width):
            raise ValueError("occupancy values do not match grid dimensions")


def empty_occupancy_map(
    grid: RasterGrid,
    *,
    initial_value: np.int8 = FREE,
) -> OccupancyMap:
    return OccupancyMap(
        grid=grid,
        values=np.full((grid.height, grid.width), initial_value, dtype=np.int8),
    )


def write_ros_occupancy_map(
    output_base: str | Path,
    occupancy_map: OccupancyMap,
) -> tuple[Path, Path]:
    """Write a ROS map_server-compatible PNG and YAML metadata pair."""

    output_base = Path(output_base)
    png_path = output_base.with_suffix(".png")
    yaml_path = output_base.with_suffix(".yaml")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    image_values = np.full(occupancy_map.values.shape, 205, dtype=np.uint8)
    image_values[occupancy_map.values == FREE] = 254
    image_values[occupancy_map.values == OCCUPIED] = 0
    Image.fromarray(np.flipud(image_values), mode="L").save(png_path)

    grid = occupancy_map.grid
    yaml_path.write_text(
        "\n".join(
            [
                f"image: {png_path.name}",
                f"resolution: {grid.resolution:.12g}",
                f"origin: [{grid.origin_x:.12g}, {grid.origin_y:.12g}, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.196",
                "mode: trinary",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return png_path, yaml_path
