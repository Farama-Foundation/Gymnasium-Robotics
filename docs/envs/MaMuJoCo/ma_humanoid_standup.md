---
firstpage:
lastpage:
---


# Humanoid Standup
<html>
	<p align="center">
		<img src="https://gymnasium.farama.org/_images/humanoid_standup.gif" alt="Humanoid Standup" width="200"/>
	</p>
</html> 

This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is [Gymansium's MuJoCo/Humanoid Standup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/).



## Action Space
The action spaces is depended on the partitioning

### if partitioning is None:
```{figure} figures/humanoid.png
    :name: humanoid
```

| Instantiate		| `env = mamujoco_v0.parallel_env("HumanoidStandup", None)`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (17,), float32)}`			|
| Part partition	| `[(abdomen_x, abdomen_y, abdomen_z, right_hip_x, right_hip_y, right_hip_z, right_knee, left_hip_x, left_hip_y, left_hip_z, left_knee, right_shoulder1, right_shoulder2, right_elbow, left_shoulder1, left_shoulder2, left_elbow,),]`	|

If partitioning, is `None` then the environment contains a single agent with the same action space as [Gymansium's MuJoCo/Humanoid Standup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/#action-space).

| Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_y                        | hinge | torque (N m) |
| 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_z                        | hinge | torque (N m) |
| 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_x                        | hinge | torque (N m) |
| 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4        | 0.4         | right_hip_x (right_thigh)        | hinge | torque (N m) |
| 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4        | 0.4         | right_hip_z (right_thigh)        | hinge | torque (N m) |
| 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4        | 0.4         | right_hip_y (right_thigh)        | hinge | torque (N m) |
| 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4        | 0.4         | right_knee                       | hinge | torque (N m) |
| 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4        | 0.4         | left_hip_x (left_thigh)          | hinge | torque (N m) |
| 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4        | 0.4         | left_hip_z (left_thigh)          | hinge | torque (N m) |
| 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4        | 0.4         | left_hip_y (left_thigh)          | hinge | torque (N m) |
| 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4        | 0.4         | left_knee                        | hinge | torque (N m) |
| 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4        | 0.4         | right_shoulder1                  | hinge | torque (N m) |
| 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4        | 0.4         | right_shoulder2                  | hinge | torque (N m) |
| 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4        | 0.4         | right_elbow                      | hinge | torque (N m) |
| 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4        | 0.4         | left_shoulder1                   | hinge | torque (N m) |
| 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4        | 0.4         | left_shoulder2                   | hinge | torque (N m) |
| 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4        | 0.4         | left_elbow                       | hinge | torque (N m) |



### if partitioning == "9|8":  # isolate upper and lower body
```{figure} figures/humanoid_9|8.png
    :name: humanoid_9|8
```

| Instantiate		| `env = mamujoco_v0.parallel_env("HumanoidStandup", "3x1")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`			|
| Number of Agents	| 2							|
| Action Spaces		| `{'agent_0': Box(-1, 1, (9,), float32), 'agent_1' : Box(-1, 1, (8,), float32)}`			|
| Part partition	| `[(abdomen_x, abdomen_y, abdomen_z, right_shoulder1, right_shoulder2, right_elbow, left_shoulder1, left_shoulder2, left_elbow,), (right_hip_x, right_hip_y, right_hip_z, right_knee, left_hip_x, left_hip_y, left_hip_z, left_knee,)]`|

The environment is partitioned in 2 parts, one part corresponding to the upper body and one part corresponding to the lower body.

#### Agent 0 action space (Upper Body)
| Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_y                        | hinge | torque (N m) |
| 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_z                        | hinge | torque (N m) |
| 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4        | 0.4         | abdomen_x                        | hinge | torque (N m) |
| 3   | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4        | 0.4         | right_shoulder1                  | hinge | torque (N m) |
| 4   | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4        | 0.4         | right_shoulder2                  | hinge | torque (N m) |
| 5   | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4        | 0.4         | right_elbow                      | hinge | torque (N m) |
| 6   | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4        | 0.4         | left_shoulder1                   | hinge | torque (N m) |
| 7   | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4        | 0.4         | left_shoulder2                   | hinge | torque (N m) |
| 8   | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4        | 0.4         | left_elbow                       | hinge | torque (N m) |

#### Agent 1 action space (Lower Body)
| Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4        | 0.4         | right_hip_x (right_thigh)        | hinge | torque (N m) |
| 1   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4        | 0.4         | right_hip_z (right_thigh)        | hinge | torque (N m) |
| 2   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4        | 0.4         | right_hip_y (right_thigh)        | hinge | torque (N m) |
| 3   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4        | 0.4         | right_knee                       | hinge | torque (N m) |
| 4   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4        | 0.4         | left_hip_x (left_thigh)          | hinge | torque (N m) |
| 5   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4        | 0.4         | left_hip_z (left_thigh)          | hinge | torque (N m) |
| 6   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4        | 0.4         | left_hip_y (left_thigh)          | hinge | torque (N m) |
| 7   | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4        | 0.4         | left_knee                        | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes all the items of the Humanoid's torso.
See more at the [Gymnasium's Humanoid Standup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Humanoid Standup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/#observation-space) reward.



## Starting state
The starting state of the environment is the as [Gymnasium's Humanoid Standup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/#starting-state).



## Episode End
All agent terminate and truncate at the same time, given the same conditions as [Gymnasium's Humanoid Standup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/#episode-end).


## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)):
	- Added/Fixed Global observations (The Humanoids's torso: `rootx`, `rooty`, `rootz`) not being observed.
	- Fixed abdomen observations ordering.
	- Added support for body observations (`cvel`, `cinert`, `cfrc_ext`)
	- Changed action ordering to be same as [Gymnasium/MuJoCo/Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/#action-space)
