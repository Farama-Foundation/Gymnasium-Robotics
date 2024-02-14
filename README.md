[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium-Robotics/main/gymrobotics-revised-text.png" width="500px"/>
</p>

This library contains a collection of Reinforcement Learning robotic environments that use the [Gymansium](https://gymnasium.farama.org/) API. The environments run with the [MuJoCo](https://mujoco.org/) physics engine and the maintained [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html).

The documentation website is at [robotics.farama.org](https://robotics.farama.org/), and we have a public discord server (which we also use to coordinate development work) that you can join here: [https://discord.gg/YymmHrvS](https://discord.gg/YymmHrvS)

## Installation

To install the Gymnasium-Robotics environments use `pip install gymnasium-robotics`

These environments also require the MuJoCo engine from Deepmind to be installed. Instructions to install the physics engine can be found at the [MuJoCo website](https://mujoco.org/) and the [MuJoCo Github repository](https://github.com/deepmind/mujoco).  

Note that the latest environment versions use the latest mujoco python bindings maintained by the MuJoCo team. If you wish to use the old versions of the environments that depend on [mujoco-py](https://github.com/openai/mujoco-py), please install this library with `pip install gymnasium-robotics[mujoco-py]`

We support and test for Python 3.8, 3.9, 3.10 and 3.11 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

## Environments

`Gymnasium-Robotics` includes the following groups of environments:

* [Fetch](https://robotics.farama.org/envs/fetch/) - A collection of environments with a 7-DoF robot arm that has to perform manipulation tasks such as Reach, Push, Slide or Pick and Place.
* [Shadow Dexterous Hand](https://robotics.farama.org/envs/shadow_dexterous_hand/) - A collection of environments with a 24-DoF anthropomorphic robotic hand that has to perform object manipulation tasks with a cube, egg-object, or pen. There are variations of these environments that also include data from 92 touch sensors in the observation space.

The [D4RL](https://github.com/Farama-Foundation/D4RL) environments are now available. These environments have been refactored and may not have the same action/observation spaces as the original, please read their documentation:

* [Maze Environments](https://robotics.farama.org/envs/maze/) - An agent has to navigate through a maze to reach certain goal position. Two different agents can be used: a 2-DoF force-controlled ball, or the classic `Ant` agent from the [Gymnasium MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/ant/). The environment can be initialized with a variety of maze shapes with increasing levels of difficulty.
* [Adroit Arm](https://robotics.farama.org/envs/adroit_hand/) - A collection of environments that use the Shadow Dexterous Hand with additional degrees of freedom for the arm movement.
The different tasks involve hammering a nail, opening a door, twirling a pen, or picking up and moving a ball.
* [Franka Kitchen](https://robotics.farama.org/envs/franka_kitchen/) - Multitask environment in which a 9-DoF Franka robot is placed in a kitchen containing several common household items. The goal of each task is to interact with the items in order to reach a desired goal configuration.

**WIP**: generate new `D4RL` environment datasets with [Minari](https://github.com/Farama-Foundation/Minari).

## Multi-goal API

The robotic environments use an extension of the core Gymansium API by inheriting from [GoalEnv](https://robotics.farama.org/content/multi-goal_api/) class. The new API forces the environments to have a dictionary observation space that contains 3 keys:

* `observation` - The actual observation of the environment
* `desired_goal` - The goal that the agent has to achieved
* `achieved_goal` - The goal that the agent has currently achieved instead. The objective of the environments is for this value to be close to `desired_goal`

This API also exposes the function of the reward, as well as the terminated and truncated signals to re-compute their values with different goals. This functionality is useful for algorithms that use Hindsight Experience Replay (HER).

The following example demonstrates how the exposed reward, terminated, and truncated functions
can be used to re-compute the values with substituted goals. The info dictionary can be used to store
additional information that may be necessary to re-compute the reward, but that is independent of the
goal, e.g. state derived from the simulation.

```python
import gymnasium as gym

env = gym.make("FetchReach-v2")
env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# The following always has to hold:
assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
assert truncated == env.compute_truncated(obs["achieved_goal"], obs["desired_goal"], info)
assert terminated == env.compute_terminated(obs["achieved_goal"], obs["desired_goal"], info)

# However goals can also be substituted:
substitute_goal = obs["achieved_goal"].copy()
substitute_reward = env.compute_reward(obs["achieved_goal"], substitute_goal, info)
substitute_terminated = env.compute_terminated(obs["achieved_goal"], substitute_goal, info)
substitute_truncated = env.compute_truncated(obs["achieved_goal"], substitute_goal, info)
```

The `GoalEnv` class can also be used for custom environments.

## Project Maintainers
Main Contributors: [Rodrigo Perez-Vicente](https://github.com/rodrigodelazcano), [Kallinteris Andreas](https://github.com/Kallinteris-Andreas), [Jet Tai](https://github.com/jjshoots) 

Maintenance for this project is also contributed by the broader Farama team: [farama.org/team](https://farama.org/team).

## Citation

If you use this in your research, please cite:
```
@software{gymnasium_robotics2023github,
  author = {Rodrigo de Lazcano and Kallinteris Andreas and Jun Jet Tai and Seungjae Ryan Lee and Jordan Terry},
  title = {Gymnasium Robotics},
  url = {http://github.com/Farama-Foundation/Gymnasium-Robotics},
  version = {1.2.4},
  year = {2023},
}
```
