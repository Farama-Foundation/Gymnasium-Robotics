import numpy as np
import gymnasium as gym

def test_reset():
    env = gym.make("AntMaze_UMaze-v3", continuing_task=True)

    for _ in range(1000):
        obs, _ = env.reset()
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        assert dist > 0.45, f"dist={dist} < 0.45"
