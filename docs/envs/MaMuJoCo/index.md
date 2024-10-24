---
firstpage:
lastpage:
---


# MaMuJoCo (Multi-Agent MuJoCo)
```{figure} figures/mamujoco.png
    :name: mamujoco
```

MaMuJoCo was introduced in ["FACMAC: Factored Multi-Agent Centralised Policy Gradients"](https://arxiv.org/abs/2003.06709).

There are 2 types of Environments, included (1) multi-agent factorizations of [Gymnasium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/) tasks and (2) new complex MuJoCo tasks meant to me solved with multi-agent Algorithms.

Gymnasium-Robotics/MaMuJoCo Represents the first, easy to use Framework for research of agent factorization.

## API
MaMuJoCo mainly uses the [PettingZoo.ParallelAPI](https://pettingzoo.farama.org/api/parallel/), but also supports a few extra functions:

```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v1.parallel_env.map_local_actions_to_global_action
```
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v1.parallel_env.map_global_action_to_local_actions
```
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v1.parallel_env.map_global_state_to_local_observations
```
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v1.parallel_env.map_local_observations_to_global_state
```
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v1.get_parts_and_edges
```

MaMuJoCo also supports the [PettingZoo.AECAPI](https://pettingzoo.farama.org/api/aec/) but does not expose extra functions.



### Arguments
```{eval-rst}
.. autofunction:: gymnasium_robotics.mamujoco_v1.parallel_env.__init__
```



## How to create new agent factorizations
MaMuJoCo-v1 not only supports the existing factorization, but also supports creating new factorizations.
### example 'Ant-v5', '8x1'
In this example, we will create an agent factorization not present in Gymnasium-Robotics/MaMuJoCo the "Ant"/'8x1', where each agent controls a single joint/action (first implemented by [safe-MaMuJoCo](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco)).

```{figure} figures/ant_8x1.png
    :name: Ant 8 way factorization
```

first we will load the graph of MaMuJoCo:
```python
>>> from gymnasium_robotics.mamujoco_v1 import get_parts_and_edges
>>> unpartioned_nodes, edges, global_nodes = get_parts_and_edges('Ant-v5', None)
```
The `unpartioned_nodes` contain the nodes of the MaMuJoCo graph.
The `edges` well, contain the edges of the graph.
And the `global_nodes` a set of observations for all agents.

To create our '8x1' partition we will need to partition the `unpartioned_nodes`:
```python
>>> unpartioned_nodes
[(hip1, ankle1, hip2, ankle2, hip3, ankle3, hip4, ankle4)]
>>> partioned_nodes = [(unpartioned_nodes[0][0],), (unpartioned_nodes[0][1],), (unpartioned_nodes[0][2],), (unpartioned_nodes[0][3],), (unpartioned_nodes[0][4],), (unpartioned_nodes[0][5],), (unpartioned_nodes[0][6],), (unpartioned_nodes[0][7],)]
>>> partioned_nodes
[(hip1,), (ankle1,), (hip2,), (ankle2,), (hip3,), (ankle3,), (hip4,), (ankle4,)]
```
Finally package the partitions and create our environment:
```python
>>> my_agent_factorization = {"partition": partioned_nodes, "edges": edges, "globals": global_nodes}
>>> gym_env = mamujoco_v1('Ant', '8x1', agent_factorization=my_agent_factorization)
```


### example 'boston dynamics spot arm' with  custom 'quadruped|arm' factorization
Here we are Factorizing the "[Boston Dynamics Spot with arm](https://bostondynamics.com/products/spot/arm/)" robot with the robot model from [Menagarie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot), into 1 agent for the locomoting quadruped component and 1 agent for the manipulator arm component.
We are using the robot model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot).

```{figure} figures/boston_dymanics_spot_arm.png
    :name: Boston Dynamics Spot Arm factorization
```

```python
from gymnasium_robotics import mamujoco_v1
from gymnasium_robotics.envs.multiagent_mujoco.obsk import Node, HyperEdge

# Define the factorization graph
freejoint = Node(
    "freejoint",
    None,
    None,
    None,
    extra_obs={
        "qpos": lambda data: data.qpos[2:7],
        "qvel": lambda data: data.qvel[:6],
    },
)
fl_hx = Node("fl_hx", -19, -19, 0)
fl_hy = Node("fl_hy", -18, -18, 1)
fl_kn = Node("fl_kn", -17, -17, 2)
fr_hx = Node("fr_hx", -16, -16, 3)
fr_hy = Node("fr_hy", -15, -15, 4)
fr_kn = Node("fr_kn", -14, -14, 5)
hl_hx = Node("hl_hx", -13, -13, 6)
hl_hy = Node("hl_hy", -12, -12, 7)
hl_kn = Node("hl_kn", -11, -11, 8)
hr_hx = Node("hr_hx", -10, -10, 9)
hr_hy = Node("hr_hy", -9, -9, 10)
hr_kn = Node("hr_kn", -8, -8, 11)
arm_sh0 = Node("arm_sh0", -7, -7, 12)
arm_sh1 = Node("arm_sh1", -6, -6, 13)
arm_el0 = Node("arm_el0", -5, -5, 14)
arm_el1 = Node("arm_el1", -4, -4, 15)
arm_wr0 = Node("arm_wr0", -3, -3, 16)
arm_wr1 = Node("arm_wr1", -2, -2, 17)
arm_f1x = Node("arm_f1x", -1, -1, 18)

parts = [
    (  # Locomoting Quadruped Component
        fl_hx,
        fl_hy,
        fl_kn,
        fr_hx,
        fr_hy,
        fr_kn,
        hl_hx,
        hl_hy,
        hl_kn,
        hr_hx,
        hr_hy,
        hr_kn,
    ),
    (  # Arm Manipulator Component
        arm_sh0,
        arm_sh1,
        arm_el0,
        arm_el1,
        arm_wr0,
        arm_wr1,
        arm_f1x,
    ),
]

edges = [
    HyperEdge(fl_hx, fl_hy, fl_kn),
    HyperEdge(fr_hx, fr_hy, fr_kn),
    HyperEdge(hl_hx, hl_hy, hl_kn),
    HyperEdge(hr_hx, hr_hy, hr_kn),
    HyperEdge(  # Main "body" connections
        fl_hx,
        fl_hy,
        fr_hx,
        fr_hy,
        hl_hx,
        hl_hy,
        hr_hx,
        hr_hy,
        arm_sh0,
        arm_sh1,
    ),
    HyperEdge(arm_sh0, arm_sh1, arm_el0, arm_el1),
    HyperEdge(arm_el0, arm_el1, arm_wr0, arm_wr1),
    HyperEdge(arm_wr0, arm_wr1, arm_f1x),
]

global_nodes = [freejoint]

my_agent_factorization = {"partition": parts, "edges": edges, "globals": global_nodes}
env = mamujoco_v1.parallel_env(
    "Ant",
    "quadruped|arm",
    agent_factorization=my_agent_factorization,
    xml_file="./mujoco_menagerie/boston_dynamics_spot/scene_arm.xml",
)
```
Of course, you also need to add new elements to the environment and define your task, to do something useful.



## Version History
* v1:
	- Based on `Gymnasium/MuJoCo-v5` instead of `Gymnasium/MuJoCo-v4` (https://github.com/Farama-Foundation/Gymnasium/pull/572).
	- When `factorizatoion=None`, the `env.gent_action_partitions.dummy_node` now contains `action_id` (it used to be `None`).
	- Added `map_local_observations_to_global_state` & optimized runtime performance of `map_global_state_to_local_observations`.
	- Added `gym_env` argument for using environment wrappers, also can be used to load third-party `Gymnasium.MujocoEnv` environments.
* v0: Initial version release on gymnasium, and is a fork of [the original multiagent_mujuco](https://github.com/schroederdewitt/multiagent_mujoco),
	- Based on `Gymnasium/MuJoCo-v4` instead of `Gym/MuJoCo-v2`.
	- Uses PettingZoo APIs instead of an original API.
	- Added support for custom agent factorizations.
	- Added new functions `MultiAgentMujocoEnv.map_global_action_to_local_actions`, `MultiAgentMujocoEnv.map_local_actions_to_global_action`, `MultiAgentMujocoEnv.map_global_state_to_local_observations`.



```{toctree}
:hidden:
ma_ant.md
ma_coupled_half_cheetah.md
ma_half_cheetah.md
ma_hopper.md
ma_humanoid_standup.md
ma_humanoid.md
ma_multiagentswimmer.md
ma_reacher.md
ma_swimmer.md
ma_pusher.md
ma_walker2d.md
ma_single.md
```
