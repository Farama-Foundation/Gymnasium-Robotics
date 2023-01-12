---
firstpage:
lastpage:
---


# MaMuJoCo (Multi-Agent MuJoCo)
```{figure} figures/mamujoco.png
    :name: mamujoco
```

MaMuJoCo was introduced in ["FACMAC: Factored Multi-Agent Centralised Policy Gradients"](https://arxiv.org/abs/2003.06709).

There are 2 types of Environments, included (1) multi-agent factorizations of [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/) tasks and (2) new complex MuJoCo tasks meant to me solved with multi-agent Algorithms.

Gymansium-Robotics/MaMuJoCo Represents the first, easy to use Framework for research of agent factorization.


The `mamujoco` framework is not included in the current `1.2.0` release of `Gymnasium-Robotics` since we are performing some evaluation tests. If you want to try the current implementation of these environments please install them from source:

```sh
git clone https://github.com/Farama-Foundation/Gymnasium-Robotics.git
cd Gymnasium-Robotics
pip install -e.
```

## API
MaMuJoCo mainly uses the [PettingZoo.ParallelAPI](https://pettingzoo.farama.org/api/parallel/), but also supports a few extra functions:

```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v0.parallel_env.map_local_actions_to_global_action
```

```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v0.parallel_env.map_global_action_to_local_actions
```
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v0.parallel_env.map_global_state_to_local_observations
```
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v0.parallel_env.map_local_observation_to_global_state
```
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v0.get_parts_and_edges
```

MaMuJoCo also supports the [PettingZoo.AECAPI](https://pettingzoo.farama.org/api/aec/) but does not expose extra functions.



### Arguments
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v0.parallel_env.__init__
```



## How to create new agent factorizations 
### example 'Ant-v4', '8x1'
In this example, we will create an agent factorization not present in Gymnasium-Robotics/MaMuJoCo the "Ant"/'8x1', where each agent controls a single joint/action (first implemented by [safe-MaMuJoCo](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco)).

first we will load the graph of MaMuJoCo:
```python
>>> from gymnasium_robotics.mamujoco_v0 import get_parts_and_edges
>>> unpartioned_nodes, edges, global_nodes = get_parts_and_edges('Ant-v4', None)
```
The `unpartioned_nodes` contain the nodes of the MaMuJoCo graph.
The `edges` well, contain the edges of the graph.
And the `global_nodes` a set of observations for all agents.

To create our '8x1' partition we will need to partition the `unpartioned_nodes`:
```python
>>> unpartioned_nodes
[(hip1, ankle1, hip2, ankle2, hip3, ankle3, hip4, ankle4)]
>>> partioned_nodes = [(unpartioned_nodes[0][0],), (unpartioned_nodes[0][1],), (unpartioned_nodes[0][2],), (unpartioned_nodes[0][3],), (unpartioned_nodes[0][4],), (unpartioned_nodes[0][5],), (unpartioned_nodes[0][6],), (unpartioned_nodes[0][7],)]>>> partioned_nodes
>>> partioned_nodes
[(hip1,), (ankle1,), (hip2,), (ankle2,), (hip3,), (ankle3,), (hip4,), (ankle4,)]
```
Finally package the partitions and create our environment:
```python
>>> my_agent_factorization = {"partition": partioned_nodes, "edges": edges, "globals": global_nodes}
>>> gym_env = mamujoco_v0('Ant', '8x1', agent_factorization=my_agent_factorization)
```

## Version History
v0: Initial version release, uses [Gymnasium.MuJoCo-v4](https://gymnasium.farama.org/environments/mujoco/), and is a fork of [the original multiagent_mujuco](https://github.com/schroederdewitt/multiagent_mujoco)

```{toctree}
:hidden:
ma_ant.md
ma_coupled_half_cheetah.md
ma_half_cheetah.md
ma_hopper.md
ma_humanoid_standup.md
ma_humanoid.md
ma_reacher.md
ma_swimmer.md
ma_pusher.md
ma_walker2d.md
ma_single.md
```
