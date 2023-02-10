---
firstpage:
lastpage:
---


# Swimmer
<html>
	<p align="center">
		<img src="https://gymnasium.farama.org/_images/swimmer.gif" alt="Swimmer" width="200"/>
	</p>
</html> 

This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is [Gymansium's MuJoCo/Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/).



## Action Space
The action spaces is depended on the partitioning

### if partitioning is None:
```{figure} figures/swimmer.png
    :name: swimmer
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Swimmer", None)`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (2,), float32)}`			|
| Part partition	| `(joint0, joint1,)`	|

If partitioning, is `None` then the environment contains a single agent with the same action space as [Gymansium's MuJoCo/Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#action-space).

| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the first rotor  | -1          | 1           | motor1_rot                       | hinge | torque (N m) |
| 1   | Torque applied on the second rotor | -1          | 1           | motor2_rot                       | hinge | torque (N m) |



### if partitioning == "2x1":  # isolate upper and lower body
```{figure} figures/swimmer_2x1.png
    :name: swimmer_2x1
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Swimmer", "2x1")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`			|
| Number of Agents	| 2							|
| Action Spaces		| `{'agent_0': Box(-1, 1, (1,), float32), 'agent_1' : Box(-1, 1, (1,), float32)}`			|
| Part partition	| `[(joint0,), (joint1,)]`|

The environment is partitioned in 2 parts, one part corresponding to the first joint, and one part corresponding to the second joint.

#### Agent 0 action space
| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the first rotor  | -1          | 1           | motor1_rot                       | hinge | torque (N m) |
#### Agent 1 action space
| Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the second rotor | -1          | 1           | motor2_rot                       | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position and velocity items of the swimmer's tip.
See more at the [Gymnasium's Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#observation-space) reward.



## Starting state
The starting state of the environment is the as [Gymnasium's Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#starting-state).



## Episode End
All agent terminate and truncate at the same time, given the same conditions as [Gymnasium's Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/#episode-end).


## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)):
	- Added/Fixed Global observations (The Swimmer's front tip: `free_body_rot`) not being observed.

