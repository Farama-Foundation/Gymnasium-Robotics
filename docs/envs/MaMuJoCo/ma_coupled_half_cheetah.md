---
firstpage:
lastpage:
---


# Coupled Half Cheetah
This Environment is one of the new environments introduced with MaMuJoCo.
The environment consists of 2 half cheetahs coupled by an elastic tendon.



## Action Space
The action spaces is depended on the partitioning.

### if partitioning is None:
```{figure} figures/coupled_half_cheetah.png
    :name: coupled_half_cheetah
```

| Instantiate		| `env = mamujoco_v0.parallel_env("CoupledHalfCheetah", None)`	|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0']`					|
| Number of Agents	| 1							|
| Action Spaces		| `{'agent_0' : Box(-1, 1, (12,), float32)}`			|
| Part partition	| `(bfoot0, bshin0, bthigh0, ffoot0, fshin0, fthigh0, bfoot1, bshin1, bthigh1, ffoot1, fshin1, fthigh1,),`	|

If partitioning, is `None`, then the environment contains a single agent with the same action space.

| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back thigh rotor of the first cheetah   | -1          | 1           | bthigh0                          | hinge | torque (N m) |
| 1   | Torque applied on the back shin rotor of the first cheetah    | -1          | 1           | bshin0                           | hinge | torque (N m) |
| 2   | Torque applied on the back foot rotor of the first cheetah    | -1          | 1           | bfoot0                           | hinge | torque (N m) |
| 3   | Torque applied on the front thigh rotor of the first cheetah  | -1          | 1           | fthigh0                          | hinge | torque (N m) |
| 4   | Torque applied on the front shin rotor of the first cheetah   | -1          | 1           | fshin0                           | hinge | torque (N m) |
| 5   | Torque applied on the front foot rotor of the first cheetah   | -1          | 1           | ffoot0                           | hinge | torque (N m) |
| 6   | Torque applied on the back thigh rotor of the second cheetah  | -1          | 1           | bthigh1                          | hinge | torque (N m) |
| 7   | Torque applied on the back shin rotor of the second cheetah   | -1          | 1           | bshin1                           | hinge | torque (N m) |
| 8   | Torque applied on the back foot rotor of the second cheetah   | -1          | 1           | bfoot1                           | hinge | torque (N m) |
| 9   | Torque applied on the front thigh rotor of the second cheetah | -1          | 1           | fthigh1                          | hinge | torque (N m) |
| 10  | Torque applied on the front shin rotor of the second cheetah  | -1          | 1           | fshin1                           | hinge | torque (N m) |
| 11  | Torque applied on the front foot rotor of the second cheetah  | -1          | 1           | ffoot1                           | hinge | torque (N m) |



### if partitioning == "1p1":  # isolate the cheetahs
```{figure} figures/coupled_half_cheetah_1p1.png
    :name: coupled_half_cheetah_1p1
```

| Instantiate		| `env = mamujoco_v0.parallel_env("CoupledHalfCheetah", "1p1")`|
|-----------------------|------------------------------------------------------|
| Agents		| `agents= ['agent_0', 'agent_1']`			|
| Number of Agents	| 2							|
| Action Spaces		| `{'agent_0': Box(-1, 1, (6,), float32), 'agent_1' : Box(-1, 1, (6,), float32)}`			|
| Part partition	| `[(bfoot0, bshin0, bthigh0, ffoot0, fshin0, fthigh0), (bfoot1, bshin1, bthigh1, ffoot1, fshin1, fthigh1),]`|

The environment is partitioned in 2 parts, one part corresponding to the first cheetah and second part corresponding to the second cheetah.

#### Agent 0 action space (first cheetah)
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back thigh rotor of the first cheetah   | -1          | 1           | bthigh0                          | hinge | torque (N m) |
| 1   | Torque applied on the back shin rotor of the first cheetah    | -1          | 1           | bshin0                           | hinge | torque (N m) |
| 2   | Torque applied on the back foot rotor of the first cheetah    | -1          | 1           | bfoot0                           | hinge | torque (N m) |
| 3   | Torque applied on the front thigh rotor of the first cheetah  | -1          | 1           | fthigh0                          | hinge | torque (N m) |
| 4   | Torque applied on the front shin rotor of the first cheetah   | -1          | 1           | fshin0                           | hinge | torque (N m) |
| 5   | Torque applied on the front foot rotor of the first cheetah   | -1          | 1           | ffoot0                           | hinge | torque (N m) |

#### Agent 1 action space (second cheetah)
| Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
| --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
| 0   | Torque applied on the back thigh rotor of the second cheetah  | -1          | 1           | bthigh1                          | hinge | torque (N m) |
| 1   | Torque applied on the back shin rotor of the second cheetah   | -1          | 1           | bshin1                           | hinge | torque (N m) |
| 2   | Torque applied on the back foot rotor of the second cheetah   | -1          | 1           | bfoot1                           | hinge | torque (N m) |
| 3   | Torque applied on the front thigh rotor of the second cheetah | -1          | 1           | fthigh1                          | hinge | torque (N m) |
| 4   | Torque applied on the front shin rotor of the second cheetah  | -1          | 1           | fshin1                           | hinge | torque (N m) |
| 5   | Torque applied on the front foot rotor of the second cheetah  | -1          | 1           | ffoot1                           | hinge | torque (N m) |



## Observation Space
Besides the local observation of each agent (which depend on their parts of the agent, the observation categories and the observation depth), each agent also observes the position and velocity items in each cheetah's top.



## Rewards
All agents receive the same average reward of each cheetah.



## Starting state
The starting state of the environment is the as [Gymnasium's Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#starting-state) (but with 2 cheetahs).



## Episode End
All agent terminate and truncate at the same time, given the same conditions as [Gymnasium's Half Cheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#episode-end).



## Version History
- v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of the original MaMuJoCo [schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco).
Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)):
	- Fixed action mapping of the second cheetah (It would previously not work)
	- Fixed tendon Jacobean observations
	- Added/Fixed Global observations (The Cheetahes's front tips: `rootx`s, `rooty`s, `rootz`s) not being observed.
 	- Improved node naming
	- Changed action ordering to be same as [Gymnasium/MuJoCo/HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/#action-space)

