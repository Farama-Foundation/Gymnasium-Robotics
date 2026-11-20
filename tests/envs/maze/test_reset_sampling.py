"""Regression coverage for maze reset location sampling."""

import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


@pytest.mark.parametrize(
    "env_id, seed",
    [
        ("PointMaze_UMaze-v3", 0),
        ("AntMaze_UMaze-v3", 6),
        ("AntMaze_UMaze-v4", 0),
        ("AntMaze_UMaze-v5", 0),
    ],
)
def test_reset_rejects_maze_without_distant_spawn(env_id, seed):
    """An impossible reset must fail promptly rather than sample forever."""
    code = f"""
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
env = gym.make({env_id!r}, maze_map=[[1, 1, 1], [1, 0, 1], [1, 1, 1]])
print("native maze constructed", flush=True)
try:
    try:
        env.reset(seed={seed})
    except ValueError as error:
        assert "reset location" in str(error)
    else:
        raise AssertionError("An impossible automatic reset was accepted")
finally:
    env.close()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Reset did not finish: {error.stdout!r}")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("env_id", ["PointMaze_UMaze-v3", "AntMaze_UMaze-v3", "AntMaze_UMaze-v4", "AntMaze_UMaze-v5"])
def test_reset_with_valid_custom_maze_is_repeatable(env_id):
    """Normal reset sampling retains its seeded results and separation."""
    maze_map = [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
    with gym.make(env_id, maze_map=maze_map) as env:
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        assert np.linalg.norm(first["achieved_goal"] - first["desired_goal"]) > 0.45


def test_explicit_reset_cell_can_bypass_automatic_separation():
    """Explicit reset coordinates remain available for a single-cell maze."""
    with gym.make("PointMaze_UMaze-v3", maze_map=[[1, 1, 1], [1, 0, 1], [1, 1, 1]]) as env:
        obs, _ = env.reset(seed=0, options={"reset_cell": [1, 1]})
        assert obs["observation"].shape == (4,)
