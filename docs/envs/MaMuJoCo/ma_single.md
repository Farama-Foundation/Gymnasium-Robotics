---
firstpage:
lastpage:
---


# Single Action Environments
MaMuJoCo also support single action [Gymansium/MuJoCo/](https://gymnasium.farama.org/environments/mujoco/) environment such as [Gymnasium/Mujoco/InvertedPendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) and [Gymnasium/Mujoco/InvertedDoublePendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)

And can be instantiated (without a partition)
`env = mamujoco_v0.parallel_env("InvertedDoublePendulum", None)`
`env = mamujoco_v0.parallel_env("InvertedPendulum", None)`

In which case, they simply are the same environments with a single agent using the `PettingZoo` APIs.


## Action Space
The action spaces is depended on the partitioning

## Observation Space
The agent receive the same [Gymnasium's Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/#observation-space) reward.



## Rewards
The agent receive the same reward as the single agent Gymnasium environment.



## Starting state
The starting state of the environment is the same as single agent Gymnasium environment.



## Episode End
The agent terminates and truncates at the same time, given the same conditions as the single agent Gymnasium environment.



## Version History
v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of [the original multiagent_mujuco](https://github.com/schroederdewitt/multiagent_mujoco).
No Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)).

