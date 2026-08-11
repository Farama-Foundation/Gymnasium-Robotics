# Gymnasium-Robotics with Adroit Binary Environments

Adroit Hand manipulation tasks with binary reward signals (0 on success, -1 otherwise) using Gymnasium API and MuJoCo.

## Environments

- **AdroitHandDoorBinary-v1** - Open a door
- **AdroitHandHammerBinary-v1** - Hammer a nail
- **AdroitHandPenBinary-v1** - Twirl a pen
- **AdroitHandRelocateBinary-v1** - Pick and relocate a ball

## Quick Start

```bash
pip install git+https://github.com/thanhtnguyen10/Gymnasium-Robotics.git
```

Requires [MuJoCo](https://mujoco.org/) and [Gymnasium](https://gymnasium.farama.org/).

## Example

```python
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("AdroitHandDoorBinary-v1")
observation, info = env.reset()
```This repository adds the Adroit Binary environments
