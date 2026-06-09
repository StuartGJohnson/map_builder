from __future__ import annotations

import argparse

from map_builder.pcd_filter import filter_point_cloud_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voxelize a PLY point cloud and remove outliers."
    )
    parser.add_argument("input", help="input PLY point cloud")
    parser.add_argument("output", help="output PLY point cloud")
    args = parser.parse_args()

    result = filter_point_cloud_file(args.input, args.output)
    print(
        "wrote {output}: input={input}, voxel={voxel}, "
        "statistical={statistical}, radius={radius}".format(
            output=args.output,
            input=result.input_point_count,
            voxel=result.voxel_point_count,
            statistical=result.statistical_point_count,
            radius=result.radius_point_count,
        )
    )


if __name__ == "__main__":
    main()
