import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from map_builder.pcd_preproc import (
    _floor_transform,
    preprocess_point_cloud_file,
)
from map_builder.test_support import metashape_functional_test


@metashape_functional_test
class TestPointCloudPreprocessing(unittest.TestCase):
    def test_processes_metashape_point_cloud(self):
        repo_root = Path(__file__).absolute().parents[1]
        input_path = repo_root / "data" / "metashape_pointcloud.ply"
        self.assertTrue(input_path.exists(), f"missing test fixture: {input_path}")

        output_dir = repo_root / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "metashape_pointcloud_preproc.ply"
        result = preprocess_point_cloud_file(
            input_path,
            output_path,
            max_ransac_points=50_000,
            floor_iterations=500,
            ceiling_iterations=500,
            target_ceiling_height=2.4384,
        )

        self.assertTrue(output_path.exists())
        self.assertEqual(result.point_count, 14_507_700)
        self.assertGreater(result.scale, 0.0)
        self.assertAlmostEqual(
            result.ceiling_height_before_scale * result.scale,
            2.4384,
            places=6,
        )

        output_pcd = o3d.t.io.read_point_cloud(str(output_path))
        self.assertEqual(output_pcd.point.positions.shape[0], result.point_count)
        self.assertIn("colors", output_pcd.point)
        self.assertIn("normals", output_pcd.point)
        self.assertIn("class", output_pcd.point)

        input_pcd = o3d.t.io.read_point_cloud(str(input_path))
        input_positions = input_pcd.point.positions.numpy()
        output_positions = output_pcd.point.positions.numpy()
        origin, basis = _floor_transform(np.asarray(result.floor_plane))
        check_indices = np.array([0, result.point_count // 2, result.point_count - 1])
        expected = (
            (input_positions[check_indices].astype(np.float64) - origin)
            @ basis
            * result.scale
        )
        np.testing.assert_allclose(
            output_positions[check_indices],
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
        homogeneous = np.column_stack(
            (
                input_positions[check_indices].astype(np.float64),
                np.ones(check_indices.shape[0]),
            )
        )
        matrix_expected = (
            np.asarray(result.source_to_processed_transform) @ homogeneous.T
        ).T[:, :3]
        np.testing.assert_allclose(matrix_expected, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
