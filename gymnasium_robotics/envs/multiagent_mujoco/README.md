# This is a fork of the orignal [MaMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco)


```diff
+ Now Uses the standard `PettingZoo`'s `ParallelEnv` API
+ Now Uses a modern verion of `Gymnasium` (10.8v -> 26.3v) 
+ Now Uses newer mujoco bindings
+ Fixed plenty of bugs
```

# Multi-Agent Mujoco
Benchmark for Continuous Multi-Agent Robotic Control, based on OpenAI's Mujoco Gym environments.

<img src="https://github.com/schroederdewitt/multiagent_mujoco/blob/master/docs/images/mamujoco.jpg" width="900" height="384">

Described in the paper [Deep Multi-Agent Reinforcement Learning for Decentralized Continuous Cooperative Control](https://arxiv.org/abs/2003.06709) by Christian Schroeder de Witt, Bei Peng, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Böhmer and Shimon Whiteson, Torr Vision Group and Whiteson Research Lab, University of Oxford, 2020

# Installation

```
git clone https://github.com/Kallinteris-Andreas/multiagent_mujoco_v4.git
cd multiagent_mujoco_v4 
pip install .
```

# Example

```python
import numpy
from multiagent_mujoco.mujoco_multi import MaMuJoCo

if __name__ == "__main__":
    #env = MaMuJoCo(scenario='InvertedPendulum', agent_conf=None, agent_obsk=1, render_mode='human')
    env = MaMuJoCo(scenario='Swimmer', agent_conf='2x1', agent_obsk=1, render_mode='human')
    #env = MaMuJoCo(scenario='Ant', agent_conf='2x4', agent_obsk=1, render_mode='human')
    #env = MaMuJoCo(scenario='manyagent_swimmer', agent_conf='10x2', agent_obsk=1, render_mode='human')
    #env = MaMuJoCo(scenario='coupled_half_cheetah', agent_conf='1p1', agent_obsk=1, render_mode='human')
    
    n_episodes = 1
    debug_step = 0

    for e in range(n_episodes):
        obs = env.reset()
        terminated = {'0': False}
        truncated = {'0': False}
        episode_reward = 0

        while not terminated['0'] and not truncated['0']:
            state = env.state()

            actions = {}
            for agent_id in env.agents:
                avail_actions = env.action_space(agent_id)
                action = numpy.random.uniform(avail_actions.low[0], avail_actions.high[0], avail_actions.shape[0])
                actions[str(agent_id)] = action

            obs, reward, terminated, truncated, info = env.step(actions)

            episode_reward += reward['0']

        print("Total reward in episode {} = {}".format(e, episode_reward))
    env.close()
```

# Documentation

## Environment config

* *scenario*: Determines the underlying single-agent OpenAI Gym Mujoco environment
* *agent_conf*: Determines the partitioning (see in Environment section below), fixed by n_agents x motors_per_agent
* *agent_obsk*: Determines up to which connection distance k agents will be able to form observations (0: agents can only observe the state of their own joints and bodies, 1: agents can observe their immediate neighbour's joints and bodies).
* *k_categories* (NOT IMPLEMENTED): A string describing which properties are observable at which connection distance as comma-separated lists separated by vertical bars. For example, "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos" means k=0 can observe properties qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator and k>=1 (i.e. immediate and more distant neighbours) can be observed through property qpos. Note: If a property requested is not available for a given agent, it will be silently omitted.
* *global_categories* (NOT IMPLEMENTED): Same as k_categories, but concerns some global properties that are otherwise not observed by any of the agents. Switched off by default (i.e. agents have no non-local observations).

# Extending Tasks

Tasks can be trivially extended by adding entries in src/multiagent_mujoco/obsk.py.

## Task configuration

Unless stated otherwise, all the parameters given below are to be used with ```.multiagent_mujoco.MujocoMulti```.

### 2-Agent Ant

```python
scenario="Ant-v2"
agent_conf="2x4"
agent_obsk=1
```

### 2-Agent Ant Diag

```python
scenario="Ant-v2"
agent_conf="2x4d"
agent_obsk=1
```

### 4-Agent Ant

```python
scenario="Ant-v2"
agent_conf="4x2"
agent_obsk=1
```

### 2-Agent HalfCheetah

```python
scenario="HalfCheetah-v2"
agent_conf="2x3"
agent_obsk=1
```

### 6-Agent HalfCheetah

```python
scenario="HalfCheetah-v2"
agent_conf="6x1"
agent_obsk=1
```

### 3-Agent Hopper

```python
scenario="Hopper-v2"
agent_conf="3x1"
agent_obsk=1
```

### 2-Agent Humanoid

```python
scenario="Humanoid-v2"
agent_conf="9|8"
agent_obsk=1
```

### 2-Agent HumanoidStandup

```python
scenario="HumanoidStandup-v2"
agent_conf="9|8"
agent_obsk=1
```

### 2-Agent Reacher

```python
scenario="Reacher-v2"
agent_conf="2x1"
agent_obsk=1
```

### 2-Agent Swimmer

```python
scenario="Swimmer-v2"
agent_conf="2x1"
agent_obsk=1
```

### 2-Agent Walker

```python
scenario="Walker2d-v2"
agent_conf="2x3"
agent_obsk=1
```


### Manyagent Swimmer

```python
scenario="manyagent_swimmer"
agent_conf="10x2"
agent_obsk=1
```


### Manyagent Ant

```python
scenario="manyagent_ant"
agent_conf="2x3"
agent_obsk=1
```

### Coupled HalfCheetah (NEW!)

```python
scenario="coupled_half_cheetah"
agent_conf="1p1"
agent_obsk=1
```

```CoupledHalfCheetah``` features two separate HalfCheetah agents coupled by an elastic tendon. You can add more tendons or novel coupled scenarios by 

1. Creating a new Gym environment to define the reward function of the coupled scenario (consult ```coupled_half_cheetah.py```)
2. Create a new Mujoco environment XML file to insert agents and tendons (see ```assets/coupled_half_cheetah.xml```)
3. Register your env as a scenario in the MujocoMulti environment (only if you need special default observability params)
