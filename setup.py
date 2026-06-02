import os
from glob import glob

from setuptools import find_packages, setup

package_name = "px4_slam"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Graham Clifford",
    maintainer_email="gjcliff@gmail.com",
    description="estimating the state of things",
    license="MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "state_estimation = px4_slam.state_estimation:main",
            "backend = px4_slam.backend:main",
            "super_flow = px4_slam.super_flow:main",
            "square_flier = px4_slam.square_flier:main",
        ],
    },
)
