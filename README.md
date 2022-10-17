[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium-Robotics/main/gymrobotics-revised-text.png" width="500px"/>
</p>

This library contains a collection of Reinforcement Learning robotic environments that use the [Gymansium](https://gymnasium.farama.org/) API. The environments run with the [MuJoCo](https://mujoco.org/) physics engine and the maintained [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html).

The documentation website is at [robotics.farama.org](https://robotics.farama.org/), and we have a public discord server (which we also use to coordinate development work) that you can join here: [https://discord.gg/YymmHrvS](https://discord.gg/YymmHrvS)

## Environments

Gymnasium-Robotics includes the following groups of environments:

* [Fetch](https://robotics.farama.org/envs/#fetch-environments) - A collection of environments with a 7-DoF robot arm that has to perform manipulation tasks such as Reach, Push, Slide or Pick and Place.
* [Shadow Dexterous Hand](https://robotics.farama.org/envs/#shadow-dexterous-hand-environments) - A collection of environments with a 24-DoF anthropomorphic robotic hand that has to perform object manipulation tasks with a cube, egg-object, or pen.
* [Shadow Hand Dexterous with Touch Sensors](https://robotics.farama.org/envs/#hand-environments-with-touch-sensors) - Variations of the `Shadow Dexterous Hand` environments that include data from 92 touch sensors in the observation space.

## Installation

To install the Gymnasium-Robotics environments use `pip install gymnasium-robotics`

These environments also require the MuJoCo engine from Deepmind to be installed. Instructions to install the physics engine can be found at the [MuJoCo website](https://mujoco.org/) and the [MuJoCo Github repository](https://github.com/deepmind/mujoco).  

Note that the latest environment versions use the latest mujoco python bindings maintained by the MuJoCo team. If you wish to use the old versions of the environments that depende on [mujoco-py](https://github.com/openai/mujoco-py) please install this library with `pip install gymnasium-robotics[mujoco-py]`

We support and test for Python 3.7, 3.8, 3.9 and 3.10 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

## Citation

If using the `Fetch` or `Shadow Hand` environments, please cite:

```bibtex
@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}
```

To cite the `Shadow Dexterous Hand with Touch Sensors` environments, please use:

```bibtex
@article{melnik2021using,
  title={Using tactile sensing to improve the sample efficiency and performance of deep deterministic policy gradients for simulated in-hand manipulation tasks},
  author={Melnik, Andrew and Lach, Luca and Plappert, Matthias and Korthals, Timo and Haschke, Robert and Ritter, Helge},
  journal={Frontiers in Robotics and AI},
  pages={57},
  year={2021},
  publisher={Frontiers}
}
```