from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import gymnasium_robotics
from gymnasium_robotics.envs.maze.maze_v4 import MazeEnv
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv

gym.register_envs(gymnasium_robotics)


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


def test_update_goal_rolls_target_for_continuing_tasks():
    env = MazeEnv.__new__(MazeEnv)
    env.continuing_task = True
    env.reset_target = False
    env.goal = np.array([0.0, 0.0])
    env.maze = SimpleNamespace(unique_goal_locations=[0, 1])
    env.generate_target_goal = lambda: np.array([1.0, 1.0])
    env.add_xy_position_noise = lambda goal: goal
    update_calls = {"count": 0}
    env.update_target_site_pos = lambda: update_calls.__setitem__("count", update_calls["count"] + 1)

    MazeEnv.update_goal(env, np.array([0.1, 0.1]))

    np.testing.assert_array_equal(env.goal, np.array([1.0, 1.0]))
    assert update_calls["count"] == 1


def test_point_maze_step_returns_post_rollover_goal():
    env = PointMazeEnv.__new__(PointMazeEnv)
    env.goal = np.array([0.0, 0.0])
    env.point_env = SimpleNamespace(
        step=lambda action: (np.array([0.0, 0.0, 0.5, -0.25]), None, None, None, {})
    )
    env.compute_reward = lambda achieved_goal, desired_goal, info: float(
        np.allclose(desired_goal, [1.0, 1.0])
    )
    env.compute_terminated = lambda achieved_goal, desired_goal, info: bool(
        np.allclose(desired_goal, [1.0, 1.0])
    )
    env.compute_truncated = lambda achieved_goal, desired_goal, info: False
    env.update_goal = lambda achieved_goal: setattr(env, "goal", np.array([1.0, 1.0]))

    obs, reward, terminated, truncated, info = PointMazeEnv.step(env, None)

    np.testing.assert_array_equal(obs["achieved_goal"], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(obs["desired_goal"], np.array([1.0, 1.0]))
    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
