import unittest
from pathlib import Path

import open3d as o3d

from map_builder.pcd_filter import filter_point_cloud_file
from map_builder.test_support import metashape_functional_test


@metashape_functional_test
class TestPointCloudFilter(unittest.TestCase):
    def test_filters_preprocessed_metashape_point_cloud(self):
        repo_root = Path(__file__).absolute().parents[1]
        output_dir = repo_root / "test_output"
        input_path = output_dir / "metashape_pointcloud_preproc.ply"
        output_path = output_dir / "metashape_pointcloud_filtered.ply"
        self.assertTrue(
            input_path.exists(),
            f"missing preprocessing output; run test_pcd_preproc first: {input_path}",
        )

        result = filter_point_cloud_file(input_path, output_path)

        self.assertTrue(output_path.exists())
        self.assertGreater(result.input_point_count, result.voxel_point_count)
        self.assertGreaterEqual(
            result.voxel_point_count,
            result.statistical_point_count,
        )
        self.assertGreaterEqual(
            result.statistical_point_count,
            result.radius_point_count,
        )
        self.assertGreater(result.radius_point_count, 0)

        output_pcd = o3d.t.io.read_point_cloud(str(output_path))
        self.assertEqual(
            output_pcd.point.positions.shape[0],
            result.radius_point_count,
        )
        self.assertIn("colors", output_pcd.point)
        self.assertIn("normals", output_pcd.point)
        self.assertIn("class", output_pcd.point)


if __name__ == "__main__":
    unittest.main()
