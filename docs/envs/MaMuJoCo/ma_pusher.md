---
firstpage:
lastpage:
---


# Pusher
<html>
	<p align="center">
		<img src="https://gymnasium.farama.org/_images/pusher.gif" alt="Pusher" width="200"/>
	</p>
</html> 

This Environment is part of [MaMuJoCo](https://robotics.farama.org/envs/MaMuJoCo/) environments. Please read that page first for general information.
The task is [Gymansium's MuJoCo/Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/).



## Action Space
The action spaces is depended on the partitioning

### if partitioning is None:
```{figure} figures/pusher.png
    :name: pusher
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Pusher", None)`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (7,), float32)}`			|
| Part partition	| `[(r_shoulder_pan_joint, r_shoulder_lift_joint, r_upper_arm_roll_joint, r_elbow_flex_joint, r_forearm_roll_joint, r_wrist_flex_joint, r_wrist_roll_joint,),]`	|

If partitioning, is None then the environment contains a single agent with the same action space as [Gymansium's MuJoCo/Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/#action-space).

| Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0    | Rotation of the panning the shoulder                              | -2          | 2           | r_shoulder_pan_joint             | hinge | torque (N m) |
| 1    | Rotation of the shoulder lifting joint                            | -2          | 2           | r_shoulder_lift_joint            | hinge | torque (N m) |
| 2    | Rotation of the shoulder rolling joint                            | -2          | 2           | r_upper_arm_roll_joint           | hinge | torque (N m) |
| 3    | Rotation of hinge joint that flexed the elbow                     | -2          | 2           | r_elbow_flex_joint               | hinge | torque (N m) |
| 4    | Rotation of hinge that rolls the forearm                          | -2          | 2           | r_forearm_roll_joint             | hinge | torque (N m) |
| 5    | Rotation of flexing the wrist                                     | -2          | 2           | r_wrist_flex_joint               | hinge | torque (N m) |
| 6    | Rotation of rolling the wrist                                     | -2          | 2           | r_wrist_roll_joint               | hinge | torque (N m) |



### if partitioning == "3p":
```{figure} figures/pusher_3p.png
    :name: pusher_3p
```

| Instantiate		| `env = mamujoco_v0.parallel_env("Pusher", "3p")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`			|
| Number of Agents	| 3							|
| Action Spaces		| `{'agent_0': Box(-1, 1, (3,), float32), 'agent_1' : Box(-1, 1, (1,), float32), 'agent_2': Box(-1, 1, (1,), float32)}`			|
| Part partition	| `[(r_shoulder_pan_joint, r_shoulder_lift_joint, r_upper_arm_roll_joint,), (r_elbow_flex_joint,), (r_forearm_roll_joint, r_wrist_flex_joint, r_wrist_roll_joint),]`|

The environment is partitioned in 3 parts, one part corresponding to the shoulder, one part corresponding to the elbow, and one part to the wrist.

#### Agent 0 action space (Shoulder)
| Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0    | Rotation of the panning the shoulder                              | -2          | 2           | r_shoulder_pan_joint             | hinge | torque (N m) |
| 1    | Rotation of the shoulder lifting joint                            | -2          | 2           | r_shoulder_lift_joint            | hinge | torque (N m) |
| 2    | Rotation of the shoulder rolling joint                            | -2          | 2           | r_upper_arm_roll_joint           | hinge | torque (N m) |

#### Agent 1 action space (Elbow)
| Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0    | Rotation of hinge joint that flexed the elbow                     | -2          | 2           | r_elbow_flex_joint               | hinge | torque (N m) |

#### Agent 2 action space (Wrist)
| Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
|-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
| 0    | Rotation of hinge that rolls the forearm                          | -2          | 2           | r_forearm_roll_joint             | hinge | torque (N m) |
| 1    | Rotation of flexing the wrist                                     | -2          | 2           | r_wrist_flex_joint               | hinge | torque (N m) |
| 2    | Rotation of rolling the wrist                                     | -2          | 2           | r_wrist_roll_joint               | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position of the pusher's object and the position of the goal.
See more at the [Gymnasium's Pusher](https://gymnasium.farama.org/environments/mujoco/reacher/#observation-space).



## Rewards
All agents receive the same [Gymnasium's Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/#observation-space) reward.



## Starting state
The starting state of the environment is the as [Gymnasium's Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/#starting-state).



## Episode End
All agent terminate and truncate at same time given the same conditions as [Gymnasium's Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/#episode-end).


## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), first implemented here.
