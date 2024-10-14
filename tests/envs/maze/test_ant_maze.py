import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_reset(version):
    """Check that AntMaze does not reset into a success state."""
    env = gym.make(f"AntMaze_UMaze-{version}", continuing_task=True)

    for _ in range(1000):
        obs, info = env.reset()
        assert not info["success"]
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        assert dist > 0.45, f"dist={dist} < 0.45"
