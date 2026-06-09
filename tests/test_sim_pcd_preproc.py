import unittest
from pathlib import Path

import open3d as o3d

from map_builder.pcd_preproc import preprocess_point_cloud_file
from map_builder.sim_room import write_sim_room_fixtures


class TestSimPointCloudPreprocessing(unittest.TestCase):
    def test_processes_sim_room_point_cloud(self):
        repo_root = Path(__file__).absolute().parents[1]
        _, input_path = write_sim_room_fixtures(repo_root / "data")
        output_path = repo_root / "test_output" / "sim_room_pointcloud_preproc.ply"

        result = preprocess_point_cloud_file(
            input_path,
            output_path,
            target_ceiling_height=2.0,
            floor_iterations=200,
            ceiling_iterations=200,
        )

        output = o3d.t.io.read_point_cloud(str(output_path))
        self.assertTrue(output_path.exists())
        self.assertAlmostEqual(
            result.ceiling_height_before_scale * result.scale,
            2.0,
            places=6,
        )
        self.assertAlmostEqual(result.scale, 1.0, places=2)
        self.assertEqual(output.point.positions.shape[0], result.point_count)


if __name__ == "__main__":
    unittest.main()
