import numpy as np
import mujoco_py
import gym


env1 = gym.make("FetchSlide-v2")
env2 = gym.make("FetchSlide-v1")

env1.reset()
env2.reset()

print(env1._utils)
