from __future__ import annotations

import argparse

from map_builder.mesh_map_cli import add_common_mesh_map_arguments, prepare_context_from_args
from map_builder.mesh_map_pipeline import generate_section_map, write_occupancy_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a ROS occupancy map from a horizontal mesh section."
    )
    add_common_mesh_map_arguments(parser)
    parser.add_argument(
        "--section-height",
        type=float,
        required=True,
        help="section height in processed meters",
    )
    parser.add_argument(
        "--footprint-width",
        type=float,
        help="section edge footprint width; defaults to map resolution",
    )
    args = parser.parse_args()

    context = prepare_context_from_args(args)
    occupancy = generate_section_map(
        context,
        section_height=args.section_height,
        footprint_width=args.footprint_width,
    )
    write_occupancy_artifacts(args.output, occupancy)


if __name__ == "__main__":
    main()
