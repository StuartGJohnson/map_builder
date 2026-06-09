import unittest
from pathlib import Path

import numpy as np

from map_builder.mesh_map_pipeline import (
    prepare_mesh_map_context,
    write_occupancy_artifacts,
)
from map_builder.occupancy_map import (
    FREE,
    OCCUPIED,
    UNKNOWN,
)
from map_builder.test_support import metashape_functional_test


@metashape_functional_test
class TestMeshFloorFreeFunctional(unittest.TestCase):
    def test_rasterizes_upward_floor_surfaces_as_free(self):
        repo_root = Path(__file__).absolute().parents[1]
        context = prepare_mesh_map_context(
            repo_root / "data" / "metashape_model.obj",
            repo_root / "data" / "metashape_pointcloud.ply",
            ceiling_height=2.4384,
            resolution=0.01,
            grid_padding=0.0,
            floor_voxel_scale=0.01,
            min_up_normal_z=0.5,
            preprocessing_options={
                "max_ransac_points": 50_000,
                "floor_iterations": 500,
                "ceiling_iterations": 500,
            },
        )
        occupancy = context.floor_map

        output_base = repo_root / "test_output" / "metashape_mesh_floor_free"
        _, png_path, yaml_path = write_occupancy_artifacts(output_base, occupancy)
        self.assertTrue(png_path.exists())
        self.assertTrue(yaml_path.exists())
        self.assertGreater(np.count_nonzero(occupancy.values == FREE), 0)
        self.assertGreater(np.count_nonzero(occupancy.values == UNKNOWN), 0)
        self.assertEqual(np.count_nonzero(occupancy.values == OCCUPIED), 0)


if __name__ == "__main__":
    unittest.main()
