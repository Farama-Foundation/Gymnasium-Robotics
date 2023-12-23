---
firstpage:
lastpage:
---

## Installation

To install the Gymnasium-Robotics environments use `pip install gymnasium-robotics`

These environments also require the MuJoCo engine from Deepmind to be installed. Instructions to install the physics engine can be found at the [MuJoCo website](https://mujoco.org/) and the [MuJoCo Github repository](https://github.com/deepmind/mujoco).  

Note that the latest environment versions use the latest mujoco python bindings maintained by the MuJoCo team. If you wish to use the old versions of the environments that depend on [mujoco-py](https://github.com/openai/mujoco-py), please install this library with `pip install gymnasium-robotics[mujoco-py]`

We support and test for Python 3.8, 3.9, 3.10 and 3.11 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.
