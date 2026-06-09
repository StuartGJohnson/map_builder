from __future__ import annotations

import argparse

from map_builder.pcd_preproc import preprocess_point_cloud_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align and scale a PLY point cloud using floor and ceiling planes."
    )
    parser.add_argument("input", help="input PLY point cloud")
    parser.add_argument("output", help="output PLY point cloud")
    parser.add_argument(
        "--ceiling-height",
        type=float,
        required=True,
        help="known floor-to-ceiling height in meters",
    )
    args = parser.parse_args()

    result = preprocess_point_cloud_file(
        args.input,
        args.output,
        target_ceiling_height=args.ceiling_height,
    )
    print(
        "wrote {output} ({points} points), scale={scale:.9g}, "
        "ceiling_before_scale={ceiling:.9g}".format(
            output=args.output,
            points=result.point_count,
            scale=result.scale,
            ceiling=result.ceiling_height_before_scale,
        )
    )


if __name__ == "__main__":
    main()
