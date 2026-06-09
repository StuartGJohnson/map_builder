from setuptools import find_packages, setup

package_name = "map_builder"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "numpy",
        "open3d",
        "pillow",
        "setuptools",
        "shapely",
        "trimesh",
    ],
    zip_safe=True,
    maintainer="sjohnson",
    maintainer_email="sjohnson@example.com",
    description="Point cloud preprocessing tools for ROS 2 map generation.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "pcd_preproc = map_builder.pcd_preproc_cli:main",
            "pcd_filter = map_builder.pcd_filter_cli:main",
            "mesh_section_map = map_builder.mesh_section_map_cli:main",
            "mesh_slab_map = map_builder.mesh_slab_map_cli:main",
        ],
    },
)
