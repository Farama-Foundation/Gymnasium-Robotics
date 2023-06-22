import gymnasium as gym
import numpy as np


def test_reset():
    """Check that AntMaze does not reset into a success state."""
    env = gym.make("AntMaze_UMaze-v4", continuing_task=True)

    for _ in range(1000):
        obs, info = env.reset()
        assert not info["success"]
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        assert dist > 0.45, f"dist={dist} < 0.45"
