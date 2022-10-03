from setuptools import find_packages, setup

import versioneer

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

# pytest is pinned to 7.0.1 as this is last version for python 3.6
extras = {
    "testing": [
        "pytest==7.0.1",
        "mujoco_py<2.2,>=2.1",
    ],
    "mujoco_py": ["mujoco_py<2.2,>=2.1"],
}

setup(
    name="gymnasium-robotics",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Legacy robotics environments from Gym repo",
    extras_require=extras,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Farama-Foundation/gymnasium-robotics",
    author="Farama Foundation",
    author_email="jkterry@farama.org",
    license="",
    packages=[
        package
        for package in find_packages()
        if package.startswith("gymnasium_robotics")
    ],
    zip_safe=False,
    install_requires=[
        "mujoco==2.2.2",
        "numpy>=1.18.0",
        "gymnasium>=0.26",
    ],
    package_data={
        "gymnasium_robotics": [
            "envs/assets/LICENSE.md",
            "envs/assets/fetch/*.xml",
            "envs/assets/hand/*.xml",
            "envs/assets/stls/fetch/*.stl",
            "envs/assets/stls/hand/*.stl",
            "envs/assets/textures/*.png",
        ]
    },
    entry_points={
        "gymnasium.envs": ["__root__ = gymnasium_robotics:register_robotics_envs"]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    tests_require=extras["testing"],
)
