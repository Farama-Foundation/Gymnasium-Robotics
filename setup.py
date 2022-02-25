import os.path
import sys

from setuptools import find_packages, setup

import versioneer

setup(
    name="gym-robotics",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Gym: A universal API for reinforcement learning environments.",
    url="https://github.com/Farama-Foundation/gym-robotics",
    author="Seungjae Ryan Lee",
    author_email="seungjaeryanlee@gmail.com",
    license="",
    packages=[
        package for package in find_packages() if package.startswith("gym_robotics")
    ],
    zip_safe=False,
    install_requires=[
        "numpy>=1.18.0",
        "cloudpickle>=1.2.0",
        "importlib_metadata>=4.8.1; python_version < '3.8'",
        "gym>=0.22",
    ],
    package_data={
        "gym_robotics": [
            "envs/assets/LICENSE.md",
            "envs/assets/fetch/*.xml",
            "envs/assets/hand/*.xml",
            "envs/assets/stls/fetch/*.stl",
            "envs/assets/stls/hand/*.stl",
            "envs/assets/textures/*.png",
        ]
    },
    entry_points={"gym.envs": ["__root__=gym_robotics:register_robotics_envs"]},
    tests_require=["pytest", "mock"],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
