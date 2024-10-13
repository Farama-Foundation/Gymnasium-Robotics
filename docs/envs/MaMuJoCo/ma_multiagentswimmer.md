---
firstpage:
lastpage:
---


# ManySegmentSwimmer
```{figure} figures/many_segment_swimmer.png
    :name: many_segment_swimmer
```


This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is variation of [Gymansium's MuJoCo/Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/), which instead of having 2 segments, it has configurable amount of segments.

The task was first introduced by Christian A. Schroeder de Witt in ["FACMAC: Factored Multi-Agent Centralised Policy Gradients"](https://arxiv.org/abs/2003.06709)


## Action Space
The shape of the action space depends on the partitioning. The partitioning has the following form: `${Number Of Agents}x${Number Of Segments per Agent}`

| Instantiate		| `env = mamujoco_v1.parallel_env("ManySegmentSwimmer", ${Number Of Agents}x${Number Of Segments per Agent})`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', ..., 'agent_${Number Of Agents}']`					|
| Number of Agents	| `${Number Of Agents}`						|
| Action Spaces		| `{${agents} : Box(-1, 1, (${Number Of Segments per Agent},), float32)}`			|
| Part partition	| `(joint0, joint1,)`	|

The environment is partitioned in `${Number Of Agents}` parts, with each part corresponding to `${Number Of Segments per Agent}` joints.

#### Agent 0 action space
| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the first rotor  | -1          | 1           | motor1_rot                       | hinge | torque (N m) |
| 1   | Torque applied on the second rotor | -1          | 1           | motor2_rot                       | hinge | torque (N m) |
| ... | ...                                | -1          | 1           | ...                              | hinge | torque (N m) |
| `${Number Of Segments per Agent}` | Torque applied on the agent's last rotor | -1          | 1           | motor`${Number Of Segments per Agent}`_rot                       | hinge | torque (N m) |
#### Agent 1 action space
| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the `${Number Of Segments per Agent}` rotor | -1          | 1           | ??? | hinge | torque (N m) |
| ... | ...                                | -1          | 1           | ...                              | hinge | torque (N m) |
| `${Number Of Segments per Agent}` | Torque applied on the agent's last rotor | -1          | 1           | motor`2x${Number Of Segments per Agent}`_rot                       | hinge | torque (N m) |
#### Agent ... action space
...



## Observation Space
| Observation Categories ||
|-----------------------|------------------------------------------------------|
| Default `local_categories` | `[["qpos", "qvel"], ["qpos"]]` |
| Default `global_categories` | `("qpos", "qvel")` |
| Supported observation categories | `"qpos", "qvel"` |

Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position and velocity items of the swimmer's tip.
See more at the [Gymnasium's Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#observation-space) reward.



## Starting state
The starting state of the environment is the same as [Gymnasium's Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#starting-state).



## Episode End
All agent terminate and truncate at the same time, given the same conditions as [Gymnasium's Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#episode-end).


## Version History
* v1:
	- Now based on `Gymnasium/MuJoCo-v5` instead of `Gymnasium/MuJoCo-v4` (https://github.com/Farama-Foundation/Gymnasium/pull/572).
	- Now uses the same `option.timestep` as `Gymansum/Swimmer` (0.01).
	- Updated model to work with `mujoco>=3.0.0`.
* v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
