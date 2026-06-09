from __future__ import annotations

import argparse

from map_builder.mesh_map_cli import add_common_mesh_map_arguments, prepare_context_from_args
from map_builder.mesh_map_pipeline import generate_slab_map, write_occupancy_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a ROS occupancy map from a voxelized mesh slab."
    )
    add_common_mesh_map_arguments(parser)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--z-max", type=float, required=True)
    parser.add_argument("--voxel-scale", type=float)
    args = parser.parse_args()

    context = prepare_context_from_args(args)
    occupancy = generate_slab_map(
        context,
        z_min=args.z_min,
        z_max=args.z_max,
        voxel_scale=args.voxel_scale,
    )
    write_occupancy_artifacts(args.output, occupancy)


if __name__ == "__main__":
    main()
