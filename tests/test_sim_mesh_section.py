import unittest
from pathlib import Path

import numpy as np

from map_builder.mesh_map_pipeline import (
    generate_section_map,
    prepare_mesh_map_context,
    write_occupancy_artifacts,
)
from map_builder.occupancy_map import FREE, OCCUPIED, UNKNOWN
from map_builder.sim_room import write_sim_room_fixtures


class TestSimMeshSection(unittest.TestCase):
    def test_generates_padded_sim_section_map_at_lidar_height(self):
        repo_root = Path(__file__).absolute().parents[1]
        mesh_path, pcd_path = write_sim_room_fixtures(repo_root / "data")
        context = prepare_mesh_map_context(
            mesh_path,
            pcd_path,
            ceiling_height=2.0,
            resolution=0.05,
            grid_padding=0.5,
            floor_voxel_scale=0.05,
            preprocessing_options={
                "floor_iterations": 200,
                "ceiling_iterations": 200,
            },
        )

        occupancy = generate_section_map(context, section_height=1.0)
        write_occupancy_artifacts(
            repo_root / "test_output" / "sim_room_mesh_section",
            occupancy,
        )

        self.assertGreater(np.count_nonzero(occupancy.values == UNKNOWN), 0)
        self.assertGreater(np.count_nonzero(occupancy.values == FREE), 0)
        self.assertGreater(np.count_nonzero(occupancy.values == OCCUPIED), 0)
        self.assertEqual(occupancy.values.shape, (100, 100))
        self.assertEqual(occupancy.values[0, 0], UNKNOWN)
        self.assertEqual(occupancy.values[50, 50], FREE)


if __name__ == "__main__":
    unittest.main()
