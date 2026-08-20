import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


# PointMaze is built on `maze_v4.MazeEnv`, AntMaze-v3 on the older `maze.MazeEnv`.
@pytest.mark.parametrize("env_id", ["PointMaze_UMaze-v3", "AntMaze_UMaze-v3"])
def test_compute_terminated_answers_per_goal(env_id):
    """A batch of goals must be answered row by row, the way `compute_reward` is.

    `compute_terminated` took the norm without `axis=-1`, which collapses a
    batch into the Frobenius norm of the whole array. The single answer that
    came back described no row in particular, so relabelling a batch of
    transitions — what the goal-env API exists for — got one verdict for all of
    them.
    """
    env = gym.make(env_id, continuing_task=False)
    unwrapped = env.unwrapped
    obs, _ = env.reset(seed=0)

    desired = obs["desired_goal"]
    near = desired + 0.01
    far = desired + np.array([5.0, 5.0])
    achieved = np.stack([near, far, near, far])
    desired_batch = np.repeat(desired[None], len(achieved), axis=0)

    terminated = np.asarray(
        unwrapped.compute_terminated(achieved, desired_batch, [{}] * len(achieved))
    )
    assert terminated.shape == (4,)
    assert terminated.tolist() == [True, False, True, False]

    env.close()


@pytest.mark.parametrize("env_id", ["PointMaze_UMaze-v3", "AntMaze_UMaze-v3"])
def test_compute_terminated_single_goal_is_a_bool(env_id):
    """A single pair keeps answering with a plain bool."""
    env = gym.make(env_id, continuing_task=False)
    unwrapped = env.unwrapped
    obs, _ = env.reset(seed=0)

    desired = obs["desired_goal"]
    assert unwrapped.compute_terminated(desired + 0.01, desired, {}) is True
    assert (
        unwrapped.compute_terminated(desired + np.array([5.0, 5.0]), desired, {})
        is False
    )

    env.close()
