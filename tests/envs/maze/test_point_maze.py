import gymnasium as gym
import numpy as np


def test_reset():
    """Check that PointMaze does not reset into a success state."""
    env = gym.make("PointMaze_UMaze-v3", continuing_task=True)

    for _ in range(1000):
        obs, info = env.reset()
        assert not info["success"]
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        assert dist > 0.45, f"dist={dist} < 0.45"


def test_reset_cell():
    """Check that passing the reset_cell location ensures that the agent resets in the right cell."""
    map = [
        [1, 1, 1, 1],
        [1, "r", "r", 1],
        [1, "r", "g", 1],
        [1, 1, 1, 1],
    ]
    env = gym.make("PointMaze_UMaze-v3", maze_map=map)
    obs = env.reset(options={"reset_cell": [1, 2]}, seed=42)[0]
    desired_obs = np.array([0.67929896, 0.59868401, 0, 0])
    np.testing.assert_almost_equal(desired_obs, obs["observation"], decimal=4)


def test_goal_cell():
    """Check that passing the goal_cell location ensures that the goal spawns in the right cell."""
    map = [
        [1, 1, 1, 1],
        [1, "r", "g", 1],
        [1, "g", "g", 1],
        [1, 1, 1, 1],
    ]
    env = gym.make("PointMaze_UMaze-v3", maze_map=map)
    obs = env.reset(options={"goal_cell": [2, 1]}, seed=42)[0]
    desired_goal = np.array([-0.36302198, -0.53056078])
    np.testing.assert_almost_equal(desired_goal, obs["desired_goal"], decimal=4)
