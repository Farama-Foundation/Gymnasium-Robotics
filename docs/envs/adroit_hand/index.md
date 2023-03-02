---
firstpage:
lastpage:
---

# Adroit Hand

This environments consists of a [Shadow Dexterous Hand](https://www.shadowrobot.com/) attached to a free arm. The system can have up to 30 actuated degrees of freedom. There are 4 possible
environments that can be initialized depending on the task to be solved:

* `AdroitHandDoor-v1`: The hand has to open a door with a latch.
* `AdroitHandHammer-v1`: The hand has to hammer a nail inside a board.
* `AdroitHandPen-v1`: The hand has to manipulate a pen until it achieves a desired goal position and rotation.
* `AdroitHandRelocate-v1`: The hand has to pick up a ball and move it to a target location.

A sparse reward variant of each environment is also provided.
These environments have a reward of 10.0 for achieving the target goal, and -0.1 otherwise.
They can be initialized via:

* `AdroitHandDoorSparse-v1`
* `AdroitHandHammerSparse-v1`
* `AdroitHandPenSparse-v1`
* `AdroitHandRelocateSparse-v1`

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
