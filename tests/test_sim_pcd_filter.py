import unittest
from pathlib import Path

from map_builder.pcd_filter import filter_point_cloud
from map_builder.pcd_io import read_point_cloud, write_point_cloud
from map_builder.pcd_preproc import preprocess_point_cloud
from map_builder.sim_room import write_sim_room_fixtures


class TestSimPointCloudFilter(unittest.TestCase):
    def test_filters_sim_room_point_cloud(self):
        repo_root = Path(__file__).absolute().parents[1]
        _, pcd_path = write_sim_room_fixtures(repo_root / "data")
        pcd = read_point_cloud(pcd_path)
        processed, _ = preprocess_point_cloud(
            pcd,
            target_ceiling_height=2.0,
            floor_iterations=200,
            ceiling_iterations=200,
        )

        filtered, result = filter_point_cloud(
            processed,
            voxel_size=0.05,
            statistical_neighbors=8,
            statistical_std_ratio=2.0,
            radius_nb_points=3,
            radius_search_radius=0.15,
        )
        output_path = repo_root / "test_output" / "sim_room_pointcloud_filtered.ply"
        write_point_cloud(output_path, filtered)

        self.assertTrue(output_path.exists())
        self.assertGreater(result.input_point_count, 0)
        self.assertGreater(result.radius_point_count, 0)
        self.assertLessEqual(result.radius_point_count, result.input_point_count)


if __name__ == "__main__":
    unittest.main()
