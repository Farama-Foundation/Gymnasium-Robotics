"""Setups up the Gymnasium Robotics module."""

from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Gets the minigrid version."""
    path = "gymnasium_robotics/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


# pytest is pinned to 7.0.1 as this is last version for python 3.6
extras = {
    "testing": [
        "pytest==7.0.1",
        "mujoco_py<2.2,>=2.1",
    ],
    "mujoco_py": ["mujoco_py<2.2,>=2.1"],
}

version = get_version()
header_count, long_description = get_description()

setup(
    name="gymnasium-robotics",
    version=version,
    author="Farama Foundation",
    author_email="contact@farama.org",
    description="Robotics environments for the Gymnasium repo.",
    url="https://github.com/Farama-Foundation/gymnasium-robotics",
    license="MIT",
    license_files=("LICENSE",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "Gymnasium", "RL", "AI", "Robotics"],
    python_requires=">=3.7",
    packages=[
        package
        for package in find_packages()
        if package.startswith("gymnasium_robotics")
    ],
    include_package_data=True,
    install_requires=[
        "mujoco>=2.3.1.post1",
        "numpy>=1.21.0,<1.24.0",
        "gymnasium>=0.26",
        "imageio>=2.14.1",
    ],
    package_data={
        "gymnasium_robotics": [
            "envs/assets/LICENSE.md",
            "envs/assets/adroit_hand/*.xml",
            "envs/assets/fetch/*.xml",
            "envs/assets/hand/*.xml",
            "envs/assets/point/*.xml",
            "envs/assets/stls/fetch/*.stl",
            "envs/assets/stls/hand/*.stl",
            "envs/assets/textures/*.png",
        ]
    },
    entry_points={
        "gymnasium.envs": [
            "__root__ = gymnasium_robotics.__init__:register_robotics_envs"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
    tests_require=extras["testing"],
    extras_require=extras,
)
