import gymnasium as gym
import numpy as np


def test_reset():
    """Check that PointMaze does not reset into a success state."""
    env = gym.make("PointMaze_UMaze-v3", continuing_task=True)

    for _ in range(1000):
        obs, _ = env.reset()
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        assert dist > 0.45, f"dist={dist} < 0.45"
