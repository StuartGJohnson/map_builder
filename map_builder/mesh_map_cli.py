from __future__ import annotations

import argparse

from map_builder.mesh_map_pipeline import prepare_mesh_map_context


def add_common_mesh_map_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mesh", help="input OBJ mesh")
    parser.add_argument("point_cloud", help="input PLY point cloud")
    parser.add_argument("output", help="output map basename")
    parser.add_argument(
        "--ceiling-height",
        type=float,
        required=True,
        help="known floor-to-ceiling height in meters",
    )
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=0.5)
    parser.add_argument("--z-max-floor", type=float, default=0.05)
    parser.add_argument("--floor-voxel-scale", type=float)
    parser.add_argument("--min-up-normal-z", type=float, default=0.5)


def prepare_context_from_args(args: argparse.Namespace):
    return prepare_mesh_map_context(
        args.mesh,
        args.point_cloud,
        ceiling_height=args.ceiling_height,
        resolution=args.resolution,
        grid_padding=args.padding,
        z_max_floor=args.z_max_floor,
        floor_voxel_scale=args.floor_voxel_scale,
        min_up_normal_z=args.min_up_normal_z,
    )
