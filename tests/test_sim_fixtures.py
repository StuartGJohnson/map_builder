import unittest
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from map_builder.sim_room import write_sim_room_fixtures


class TestSimFixtures(unittest.TestCase):
    def test_generates_closed_room_obj_and_ply(self):
        repo_root = Path(__file__).absolute().parents[1]
        obj_path, ply_path = write_sim_room_fixtures(repo_root / "data")

        mesh = trimesh.load_mesh(obj_path, process=False)
        pcd = o3d.t.io.read_point_cloud(str(ply_path))
        self.assertTrue(mesh.is_watertight)
        self.assertLess(mesh.volume, 0.0)
        room_center = np.array([0.0, 0.0, 1.0])
        self.assertTrue(
            np.all(
                np.einsum(
                    "ij,ij->i",
                    mesh.face_normals,
                    mesh.triangles_center - room_center,
                )
                < 0.0
            )
        )
        self.assertGreater(pcd.point.positions.shape[0], 0)
        self.assertIn("normals", pcd.point)


if __name__ == "__main__":
    unittest.main()
