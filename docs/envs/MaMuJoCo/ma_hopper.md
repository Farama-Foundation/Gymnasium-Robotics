---
firstpage:
lastpage:
---


# Hopper
<html>
	<p align="center">
		<img src="https://gymnasium.farama.org/_images/hopper.gif" alt="Hopper" width="200"/>
	</p>
</html> 

This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is [Gymansium's MuJoCo/Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/).



## Action Space
The action spaces is depended on the partitioning

### if partitioning is None:
```{figure} figures/hopper.png
    :name: hopper
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Hopper", None)`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (3,), float32)}`			|
| Part partition	| `[(thigh_joint, leg_joint, foot_joint,)]`	|

If partitioning, is `None`, then the environment contains a single agent with the same action space as [Gymansium's MuJoCo/Half_Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).


| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
| 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                        | hinge | torque (N m) |
| 2   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                       | hinge | torque (N m) |



### if partitioning == "3x1":  # each joint
```{figure} figures/hopper_3x1.png
    :name: hopper_3x1
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Hopper", "3x1")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1', 'agent_2']`			|
| Number of Agents	| 3							|
| Action Spaces		| `{Box(-1, 1, (1,), float32)}`|
| Part partition	| `[(thigh_joint,), (leg_joint,), (foot_joint,)]`|

The environment is partitioned in 3 parts, each part corresponding to a single joint.

#### Agent 0 action space
| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
#### Agent 1 action space
| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                        | hinge | torque (N m) |
#### Agent 2 action space
| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                       | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position and velocity items of the hopper's top.
See more at the [Gymnasium's Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/#observation-space) reward.



## Starting state
The starting state of the environment is the as [Gymnasium's Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/#starting-state).



## Episode End
All agent terminate and truncate at same time given the same conditions as [Gymnasium's Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/#episode-end).


## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)):
	- Fixed Global observations (The Hopper's top: `rootx`, `rooty`, `rootz`) not being observed.
