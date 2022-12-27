from kitchen_env import FrankaRobot, KitchenEnv
import numpy as np


env = KitchenEnv(render_mode="human")

env.reset()

action = np.array([0.0, 0.0, 0.0, 0, 0, 0,0,0])

while True:
    env.step(action)

