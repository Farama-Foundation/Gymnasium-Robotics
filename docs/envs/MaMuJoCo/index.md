---
firstpage:
lastpage:
---


## MaMuJoCo (Multi-Agent MuJoCo)

These environments were introduced in ["FACMAC: Factored Multi-Agent Centralised Policy Gradients"](https://arxiv.org/abs/2003.06709)

There are 2 types of Environments, included (1) multi-agent factorizations of [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/) tasks and (2) new complex MuJoCo tasks meant to me solved with multi-agent Algorithms

This Represents the first, easy to use Framework for research of agent factorization

obb
[ant]

### API

MaMuJoCo uses the [PettingZoo.ParallelAPI](https://pettingzoo.farama.org/api/parallel/), but also supports a few extra functions
- MaMuJoCo.map_local_actions_to_global_action
- MaMuJoCo.map_global_action_to_local_actions
- MaMuJoCo.map_global_state_to_local_observations
- MaMuJoCo.map_local_observation_to_global_state (NOT IMPLEMENTED)
- obsk.get_parts_and_edges

```{raw} html
    :file: list.html
```

```{toctree}
:hidden:
ma_ant.md
```
