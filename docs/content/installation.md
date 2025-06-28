---
firstpage:
lastpage:
---

## Installation

To install the Gymnasium-Robotics environments use `pip install gymnasium-robotics`

These environments also require the MuJoCo engine from Deepmind to be installed. Instructions to install the physics engine can be found at the [MuJoCo website](https://mujoco.org/) and the [MuJoCo Github repository](https://github.com/deepmind/mujoco).

### Legacy environments
Note that the latest environment versions use the latest mujoco python bindings maintained by the MuJoCo team.
If you wish to use the old versions of the environments that depend on [mujoco-py](https://github.com/openai/mujoco-py)

We provide 2 ways of installing the `mujoco-py` bindings

You can install the latest version of `mujoco-py` bindings with:

```sh
pip install gymnasium-robotics[mujoco-py]
```

If you need to use older `mujoco-py` versions for your work (does not support `cython>=3`):
```bash
pip install gymnasium-robotics[mujoco-py-original]
```



We support and test for Python 3.10, 3.11, 3.12 and 3.13 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.
