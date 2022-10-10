[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<p align="center">
    <img src="readme.png" width="500px"/>
</p>

<br>

# Gymnasium-Robotics
A collection of robotics simulation environments for reinforcement learning based on the [MuJoCo](https://mujoco.org/) physics engine, and first introduced in the following [technical report](https://arxiv.org/abs/1802.09464).

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

<br>

## Fetch environments

The Fetch environments are based on the 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) arm, with a two-fingered parallel gripper attached to it. The main environment tasks are the following: 

* `FetchReach-v2`: Fetch has to move its end-effector to the desired goal position.
* `FetchPush-v2`: Fetch has to move a box by pushing it until it reaches a desired goal position.
* `FetchSlide-v2`: Fetch has to hit a puck across a long table such that it slides and comes to rest on the desired goal.
* `FetchPickAndPlace-v2`: Fetch has to pick up a box from a table using its gripper and move it to a desired goal above the table.

<p align="center"> <img src="https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/docs/img/fetchpickandplace.gif" alt="animated" width="300" height="300"/> </p>

<br>

## Shadow Dexterous Hand environments

These environments are based on the [Shadow Dexterous Hand](https://www.shadowrobot.com/), 5 which is an anthropomorphic robotic hand with 24 degrees of freedom. Of those 24 joints, 20 can be can be controlled independently whereas the remaining ones are coupled joints.

* `HandReach-v1`: ShadowHand has to reach with its thumb and a selected finger until they meet at a desired goal position above the palm.
* `HandManipulateBlock-v1`: ShadowHand has to manipulate a block until it achieves a desired goal position and rotation.
* `HandManipulateEgg-v1`: ShadowHand has to manipulate an egg until it achieves a desired goal position and rotation.
* `HandManipulatePen-v1`: ShadowHand has to manipulate a pen until it achieves a desired goal position and rotation.

<p align="center"> <img src="https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/docs/img/handblock.gif" alt="animated" width="300" height="300"/> </p>

<br>

# Hand environments with Touch Sensors

Touch sensor observations are also available in all Hand environments, with exception of `HandReach`. These environments add to the palm of the hand and the phalanges of the fingers 92 touch sensors with different recorded data depending on the environment. These touch sensors are:
- **Boolean Touch Sensor**: the observations of each touch sensor return a value of `0` if no contact is detected with and object, and `1` otherwise.
- **Continuous Touch Sensor**: the value returned by each touch sensor is a continuous value that represents the external force made by an object over the sensor.

These environments are instanceated by adding the following strings to the Hand environment id's: `_BooleanTouchSensor` or `_ContinuousTouchSensor`. For example, to add boolean touch sensors to `HandManipulateEgg-v1`, make the environment in the following way:

```python
import gymnasium as gym

env = gym.make('HandManipulateEgg_BooleanTouchSensor-v1')
```

<p align="center"> <img src="https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/docs/img/eggtouch.gif" alt="animated" width="300" height="300"/> </p>

<br>

If using these environments please also cite the following paper:

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
