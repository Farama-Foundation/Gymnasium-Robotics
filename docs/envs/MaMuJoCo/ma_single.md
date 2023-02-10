---
firstpage:
lastpage:
---


# Single Action Environments
MaMuJoCo also supports single action [Gymansium/MuJoCo/](https://gymnasium.farama.org/environments/mujoco/) environments such as [Gymnasium/Mujoco/InvertedPendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) and [Gymnasium/Mujoco/InvertedDoublePendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/).

And can be instantiated (without a partition):

`env = mamujoco_v0.parallel_env("InvertedDoublePendulum", None)`

`env = mamujoco_v0.parallel_env("InvertedPendulum", None)`

In which case, they simply are the same environments with a single agent using the `PettingZoo` APIs.

The Purpose of these is to allow researchers to debug multi-agent learning algorithms.



## Action Space
The action spaces is depended on the partitioning.

## Observation Space
The agent receives the same observations as the single agent Gymnasium environment.



## Rewards
The agent receive the same reward as the single agent Gymnasium environment.



## Starting state
The starting state of the environment is the same as single agent Gymnasium environment.



## Episode End
The agent terminates and truncates at the same time, given the same conditions as the single agent Gymnasium environment.



## Version History
v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of [the original multiagent_mujuco](https://github.com/schroederdewitt/multiagent_mujoco).
No Changes from the original `MaMuJoCo` ([schroederdewitt/multiagent_mujoco](https://github.com/schroederdewitt/multiagent_mujoco)).

