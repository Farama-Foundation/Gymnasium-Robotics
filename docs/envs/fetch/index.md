---
firstpage:
lastpage:
---

# Fetch

The Fetch environments are based on the 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) arm, with a two-fingered parallel gripper attached to it. The main environment tasks are the following:

* `FetchReach-v3`: Fetch has to move its end-effector to the desired goal position.
* `FetchPush-v3`: Fetch has to move a box by pushing it until it reaches a desired goal position.
* `FetchSlide-v3`: Fetch has to hit a puck across a long table such that it slides and comes to rest on the desired goal.
* `FetchPickAndPlace-v3`: Fetch has to pick up a box from a table using its gripper and move it to a desired goal above the table.

```{raw} html
    :file: list.html
```

## Reference

If using the `Fetch` environments, please cite:

```bibtex
@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}
```

```{toctree}
:glob:
:hidden:
./*
```
