import numpy as np
from kitchen_env import KitchenEnv

env = KitchenEnv(render_mode="human")

env.reset()

action = np.array([1, -1, 0, 0, 0.0, 0.0, 0.0, 0.0])

while True:
    for i in range(50):
        print(i)
        env.step(action)
    env.reset()
