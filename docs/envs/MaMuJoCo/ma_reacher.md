---
firstpage:
lastpage:
---


# Reacher
<html>
	<p align="center">
		<img src="https://gymnasium.farama.org/_images/reacher.gif" alt="Reacher" width="200"/>
	</p>
</html> 

This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is [Gymansium's MuJoCo/Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/).



## Action Space
The action spaces is depended on the partitioning

### if partitioning is None:
```{figure} figures/reacher.png
    :name: reacher
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Reacher", None)`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (2,), float32)}`			|
| Part partition	| `[(joint0,), (joint1,),]`	|

If partitioning, is `None`, then the environment contains a single agent with the same action space as [Gymansium's MuJoCo/Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/#action-space).

| Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
|-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
| 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1 | 1 | joint0  | hinge | torque (N m) |
| 1   |  Torque applied at the second hinge (connecting the two links)                  | -1 | 1 | joint1  | hinge | torque (N m) |



### if partitioning == "2x1":
```{figure} figures/reacher_2x1.png
    :name: reacher_2x1
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Reacher", "2x1")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`			|
| Number of Agents	| 2							|
| Action Spaces		| `{'agent_0': Box(-1, 1, (1,), float32), 'agent_1' : Box(-1, 1, (1,), float32)}`			|
| Part partition	| `[(joint0,), (joint1,)]`|

The environment is partitioned in 2 parts, one part corresponding to the first joint, and one part corresponding to the second joint.

#### Agent 0 action space
| Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
|-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
| 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1 | 1 | joint0  | hinge | torque (N m) |

#### Agent 1 action space
| Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
|-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
| 0   |  Torque applied at the second hinge (connecting the two links)                  | -1 | 1 | joint1  | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position of the reacher's target object.
See more at the [Gymnasium's Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/#observation-space) reward.



## Starting state
The starting state of the environment is the as [Gymnasium's Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/#starting-state).



## Episode End
All agent terminate and truncate at the same time, given the same conditions as [Gymnasium's Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/#episode-end).


## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)):
	- Added/Fixed Global observations (The Targets's coordinates: `targetx`, `targety`) not being observed.

