---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:titlesonly:
:caption: Introduction
content/installation
content/multi-goal_api
```

```{toctree}
:hidden:
:caption: Environments
envs/fetch
envs/shadow_dexterous_hand
envs/shadow_dexterous_hand_with_touch_sensors
```

```{toctree}
:hidden:
:caption: Development
Github <https://github.com/Farama-Foundation/Gymnasium-Robotics>
Donate <https://farama.org/donations>
Contribute to the Docs <https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/.github/PULL_REQUEST_TEMPLATE.md>
```
# Gymnasium-Robotics is a collection of robotics simulation environments for Reinforcement Learning


```{figure} img/fetchpickandplace.gif
   :alt: Fetch Pick And Place
   :width: 500
   :height: 500
```

This library contains a collection of Reinforcement Learning robotic environments that use the [Gymansium](https://gymnasium.farama.org/) API. The environments run with the [MuJoCo](https://mujoco.org/) physics engine and the maintained [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html).

The creation and interaction with the robotic environments follow the Gymnasium interface:

```{code-block} python

import gymnasium as gym
env = gym.make("FetchPickAndPlace-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```