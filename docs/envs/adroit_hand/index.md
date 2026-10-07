---
firstpage:
lastpage:
---

# Adroit Hand

This environments consists of a [Shadow Dexterous Hand](https://www.shadowrobot.com/) attached to a free arm. The system can have up to 30 actuated degrees of freedom. There are 4 possible
environments that can be initialized depending on the task to be solved:

* `AdroitHandDoor-v2`: The hand has to open a door with a latch.
* `AdroitHandHammer-v2`: The hand has to hammer a nail inside a board.
* `AdroitHandPen-v2`: The hand has to manipulate a pen until it achieves a desired goal position and rotation.
* `AdroitHandRelocate-v2`: The hand has to pick up a ball and move it to a target location.

A sparse reward variant of each environment is also provided.
These environments have a reward of 10.0 for achieving the target goal, and -0.1 otherwise.
They can be initialized via:

* `AdroitHandDoorSparse-v2`
* `AdroitHandHammerSparse-v2`
* `AdroitHandPenSparse-v2`
* `AdroitHandRelocateSparse-v2`

A binary reward variant is also provided, following the convention used by the offline-to-online
literature (for example RLPD and Cal-QL). These environments have a reward of 0.0 for achieving the
target goal and -1.0 otherwise, i.e. a per-step cost of 1 until the task is solved. Their episode
limits follow the same convention: 100 steps for the pen task and 200 for the others.
They can be initialized via:

* `AdroitHandDoorBinary-v1`
* `AdroitHandHammerBinary-v1`
* `AdroitHandPenBinary-v1`
* `AdroitHandRelocateBinary-v1`

```{raw} html
    :file: list.html
```

## Reference

These environments were first introduced in [“Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations”](https://arxiv.org/abs/1709.10087) by Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Which can be cited as follows:

```
@article{rajeswaran2017learning,
  title={Learning complex dexterous manipulation with deep reinforcement learning and demonstrations},
  author={Rajeswaran, Aravind and Kumar, Vikash and Gupta, Abhishek and Vezzani, Giulia and Schulman, John and Todorov, Emanuel and Levine, Sergey},
  journal={arXiv preprint arXiv:1709.10087},
  year={2017}
}
```

```{toctree}
:glob:
:hidden:
./*
```
