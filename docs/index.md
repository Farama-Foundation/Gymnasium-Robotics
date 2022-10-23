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
```

# Gymnasium-Robotics is a collection of robotics simulation environments for Reinforcement Learning

```{figure} img/fetchpickandplace.gif
   :alt: Fetch Pick And Place
   :width: 500
   :height: 500
```

This library contains a collection of Reinforcement Learning robotic environments that use the [Gymansium](https://gymnasium.farama.org/) API. The environments run with the [MuJoCo](https://mujoco.org/) physics engine and the maintained [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html).

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
