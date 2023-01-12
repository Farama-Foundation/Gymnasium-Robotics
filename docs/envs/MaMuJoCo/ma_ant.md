---
firstpage:
lastpage:
---


# Ant
<html>
	<p align="center">
		<img src="https://gymnasium.farama.org/_images/ant.gif" alt="ant" width="200"/>
	</p>
</html> 

This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is [Gymansium's MuJoCo/Ant](https://gymnasium.farama.org/environments/mujoco/ant/).



## Action Space
The action spaces is depended on the partitioning

### if partitioning is None:
```{figure} figures/ant.png
    :name: ant
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Ant", None)`		|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (8,), float32)}`			|
| Part partition	| `[(hip4, ankle4, hip1, ankle1, hip2, ankle2, hip3, ankle3)]`	|

If partitioning, is None then the environment contains a single agent with the same action space as [Gymansium's MuJoCo/Ant](https://gymnasium.farama.org/environments/mujoco/ant/).

| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
| 1   | Torque applied on the rotor between the back right two links      | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |
| 2   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
| 3   | Torque applied on the rotor between the front left two links      | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |
| 4   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
| 5   | Torque applied on the rotor between the front right two links     | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |
| 6   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
| 7   | Torque applied on the rotor between the back left two links       | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |

### if partitioning == "2x4":  # neighboring legs together (front and back)
```{figure} figures/ant_2x4.png
    :name: ant_2x4
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Ant", "2x4")`		|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`					|
| Number of Agents	| 2							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (4,), float32), 'agent_1' : Box(-1, 1, (4,), float32)}`|
| Part partition	| `[(hip1, ankle1, hip2, ankle2), (hip3, ankle3, hip4, ankle4)]`	|

The environment is partitioned in 2 parts, the front part (containing the front legs) and the back part (containing the back legs).

#### Agent 0 action space (front legs)
| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
| 1   | Torque applied on the rotor between the front left two links      | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |
| 2   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
| 3   | Torque applied on the rotor between the front right two links     | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |

#### Agent 1 action space (back legs)
| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
| 1   | Torque applied on the rotor between the back right two links      | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |
| 2   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
| 3   | Torque applied on the rotor between the back left two links       | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |

### if partitioning == "2x4d":  # diagonal legs together
```{figure} figures/ant_2x4d.png
    :name: ant_2x4d
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Ant", "2x4d")`		|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`					|
| Number of Agents	| 2							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (4,), float32), 'agent_1' : Box(-1, 1, (4,), float32)}`|
| Part partition	| `[(hip1, ankle1, hip4, ankle4), (hip2, ankle2, hip3, ankle3)]`	|

The environment is partitioned in 2 parts, split diagonally.
#### Agent 0 action space
| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
| 1   | Torque applied on the rotor between the front left two links      | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |
| 2   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
| 3   | Torque applied on the rotor between the back right two links      | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |
#### Agent 1 action space
| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
| 1   | Torque applied on the rotor between the front right two links     | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |
| 2   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
| 3   | Torque applied on the rotor between the back left two links       | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |

### if partitioning == "4x2":
```{figure} figures/ant_4x2.png
    :name: ant_4x2
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Ant", "4x2")`		|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1', 'agent_2', 'agent_3']`			|
| Number of Agents	| 4							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (2,), float32), 'agent_1' : Box(-1, 1, (2,), float32)}, 'agent_2' : Box(-1, 1, (2,), float32), 'agent_3' : Box(-1, 1, (2,), float32)},`|
| Part partition	| `[(hip1, ankle1), (hip2, ankle2), (hip3, ankle3), (hip4, ankle4)]`	|

The environment is partitioned in 4 parts, with each part corresponding to a leg of the ant.

#### Agent 0 action space (front left leg)
| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
| 1   | Torque applied on the rotor between the front left two links      | -1          | 1           | angle_1 (front_left_leg)         | hinge | torque (N m) |

#### Agent 1 action space (front right leg)
| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
| 1   | Torque applied on the rotor between the front right two links     | -1          | 1           | angle_2 (front_right_leg)        | hinge | torque (N m) |

#### Agent 2 action space (right left leg)
| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 2   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
| 3   | Torque applied on the rotor between the back left two links       | -1          | 1           | angle_3 (back_leg)               | hinge | torque (N m) |

#### Agent 3 action space (right back leg)
| Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
| 1   | Torque applied on the rotor between the back right two links      | -1          | 1           | angle_4 (right_back_leg)         | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position and velocity items of the ant's torso.
See more at the [Gymnasium's Ant](https://gymnasium.farama.org/environments/mujoco/ant/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Ant](https://gymnasium.farama.org/environments/mujoco/ant/#observation-space) reward.



## Starting state
The starting state of the environment is the as [Gymnasium's Ant](https://gymnasium.farama.org/environments/mujoco/ant/#starting-state).



## Episode End
All agent terminate and truncate at the same time given the same conditions as [Gymnasium's Ant](https://gymnasium.farama.org/environments/mujoco/ant/#episode-end).



## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)):
	- Fixed diagonal factorization ("2x4d") not being diagonal.
	- Fixed Global observations (The Ant's Torso: `rootx`, `rooty`, `rootz`) not being observed.
	- Changed action ordering to be same as [Gymnasium/MuJoCo/Ant](https://gymnasium.farama.org/environments/mujoco/ant/#action-space)
