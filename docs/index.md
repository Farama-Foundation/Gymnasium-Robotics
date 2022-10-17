---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:caption: Environments
envs/index
```
```{toctree}
:hidden:
:caption: Development
Github <https://github.com/Farama-Foundation/Gymnasium-Robotics>
Donate <https://farama.org/donations>
Contribute to the Docs <https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/.github/PULL_REQUEST_TEMPLATE.md>
```


# Gymnasium-Robotics is a collection of robotics simulation environments for reinforcement learning 
Gymnasium-robotics is based on the [MuJoCo](https://mujoco.org/) physics engine, and first introduced in the following [technical report](https://arxiv.org/abs/1802.09464).

Requirements:
- Python 3.7 to 3.10
- Gymnasium v0.26
- NumPy 1.18+
- Mujoco 2.2.2

If you use these environments, please cite the following paper:

```bibtex
@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}
```

## New MuJoCo Python Bindings

The latest version and future versions of the MuJoCo environments will no longer depend on `mujoco-py`. Instead the new [mujoco](https://mujoco.readthedocs.io/en/latest/python.html) python bindings will be the required dependency for future gymnasium MuJoCo environment versions. Old gymnasium MuJoCo environment versions that depend on `mujoco-py` will still be kept but unmaintained.
Dependencies for old MuJoCo environments can still be installed by `pip install gymnasium_robotics[mujoco_py]`.