---
firstpage:
lastpage:
---


# Half Cheetah
<html>
	<p align="center">
		<img src="https://gymnasium.farama.org/_images/half_cheetah.gif" alt="Half Cheetah" width="200"/>
	</p>
</html> 

This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is [Gymansium's MuJoCo/Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).



## Action Space
The action spaces is depended on the partitioning

### if partitioning is None:
```{figure} figures/half_cheetah.png
    :name: half_cheetah
```

| Instantiate		| `env = mamujoco_v0.parallel_env("HalfCheetah", None)`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (6,), float32)}`			|
| Part partition	| `[(bthigh, bshin, bfoot, fthigh, fshin, ffoot)]`	|

If partitioning, is `None`, then the environment contains a single agent with the same action space as [Gymansium's MuJoCo/Half_Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).

| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
| 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
| 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
| 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
| 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
| 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |

### if partitioning == "2x3":  # front and back 
```{figure} figures/half_cheetah_2x3.png
    :name: half_cheetah_2x3
```

| Instantiate		| `env = mamujoco_v0.parallel_env("HalfCheetah", "2x3")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`					|
| Number of Agents	| 2							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (3,), float32), 'agent_1' : Box(-1, 1, (3,), float32)}`|
| Part partition	| `[(bthigh, bshin, bfoot), (fthigh, fshin, ffoot)]`	|

The environment is partitioned in 2 parts, the front part (containing the front leg) and the back part (containing the back leg)

#### Agent 0 action space (front leg)
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
| 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
| 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |

#### Agent 1 action space (back leg)
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
| 1   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
| 2   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |

### if partitioning == "6x1":  # each joint
```{figure} figures/half_cheetah_6x1.png
    :name: half_cheetah_6x1
```

| Instantiate		| `env = mamujoco_v0.parallel_env("HalfCheetah", "6x1")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5']`			|
| Number of Agents	| 6							|
| Action Spaces		| `{Box(-1, 1, (1,), float32)}`|
| Part partition	| `[(bthigh,), (bshin,), (bfoot,), (fthigh,), (fshin,), (ffoot,)]`|

The environment is partitioned in 6 parts, each part corresponding to a single joint

#### Agent 0 action space
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
#### Agent 1 action space
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
#### Agent 2 action space
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
#### Agent 3 action space
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
#### Agent 4 action space
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
#### Agent 5 action space
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position and velocity items of the half cheetah's tip.
See more at the [Gymnasium's Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#observation-space) reward.



## Starting state
The starting state of the environment is the as [Gymnasium's Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#starting-state).



## Episode End
All agent terminate and truncate at the same time, given the same conditions as [Gymnasium's Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#episode-end).


## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)):
	- Added/Fixed Global observations (The Cheetah's front tip: `rootx`, `rooty`, `rootz`) not being observed.
	- Changed action ordering to be same as [Gymnasium/MuJoCo/HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#action-space)


