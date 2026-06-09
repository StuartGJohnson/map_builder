import unittest

import numpy as np

from map_builder.mesh_processing import _rasterize_square_footprints
from map_builder.occupancy_map import OCCUPIED, RasterGrid, empty_occupancy_map


class TestMeshRasterization(unittest.TestCase):
    def test_voxel_footprints_fill_cells_across_offset_lattices(self):
        grid = RasterGrid(
            resolution=0.01,
            origin_x=0.003,
            origin_y=0.003,
            width=10,
            height=1,
        )
        occupancy = empty_occupancy_map(grid)
        centers = np.column_stack(
            (
                np.arange(0.005, 0.105, 0.01),
                np.full(10, 0.005),
            )
        )

        _rasterize_square_footprints(
            occupancy,
            centers_xy=centers,
            footprint_size=0.01,
        )

        np.testing.assert_array_equal(
            occupancy.values,
            np.full((1, 10), OCCUPIED, dtype=np.int8),
        )

    def test_one_cell_footprint_expands_across_cell_boundary(self):
        grid = RasterGrid(
            resolution=0.05,
            origin_x=-0.10,
            origin_y=0.0,
            width=4,
            height=1,
        )
        occupancy = empty_occupancy_map(grid)

        _rasterize_square_footprints(
            occupancy,
            centers_xy=np.array([[0.0, 0.025]]),
            footprint_size=0.05,
        )

        np.testing.assert_array_equal(
            occupancy.values,
            np.array([[0, OCCUPIED, OCCUPIED, 0]], dtype=np.int8),
        )


if __name__ == "__main__":
    unittest.main()
