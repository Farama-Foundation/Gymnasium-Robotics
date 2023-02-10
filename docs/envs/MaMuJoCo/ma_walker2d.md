---
firstpage:
lastpage:
---


# Walker2d
<html>
	<p align="center">
		<img src="https://gymnasium.farama.org/_images/walker2d.gif" alt="Walker2D" width="200"/>
	</p>
</html> 

This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is [Gymansium's MuJoCo/Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/).



## Action Space
The action spaces is depended on the partitioning

### if partitioning is None:
```{figure} figures/walker2d.png
    :name: walker2d
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Walker2D", None)`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (6,), float32)}`			|
| Part partition	| `(foot_joint, leg_joint, thigh_joint, foot_left_joint, leg_left_joint, thigh_left_joint,),`	|

If partitioning, is `None` then the environment contains a single agent with the same action space as [Gymansium's MuJoCo/Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/#action-space).

| Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
| 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
| 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
| 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
| 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
| 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |



### if partitioning == "2x3":  # isolate right and left foot
```{figure} figures/walker2d_2x3.png
    :name: walker2d_2x3
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Walker2d", "2x3")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`			|
| Number of Agents	| 2							|
| Action Spaces		| `{'agent_0': Box(-1, 1, (3,), float32), 'agent_1' : Box(-1, 1, (3,), float32)}`			|
| Part partition	| `[(foot_joint, leg_joint, thigh_joint), (foot_left_joint, leg_left_joint, thigh_left_joint,),]`|

The environment is partitioned in 2 parts, one part corresponding to the right leg, and one part corresponding to the left leg.

#### Agent 0 action space (right leg)
| Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
| 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
| 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |

#### Agent 1 action space (left leg)
| Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
| 1   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
| 2   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position and velocity items of the walker's top.
See more at the [Gymnasium's Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/#observation-space) reward.



## Starting state
The starting state of the environment is the as [Gymnasium's Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/#starting-state).



## Episode End
All agent terminate and truncate at the same time given the same conditions as [Gymnasium's Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/#episode-end).



## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)):
	- Added/Fixed Global observations (The Walker's top: `rootx`, `rooty`, `rootz`) not being observed.
