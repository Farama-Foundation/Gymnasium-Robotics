import gymnasium as gym
import numpy as np

import gymnasium_robotics

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


def test_step_returns_updated_goal_after_goal_reset():
    """After reaching the goal in a continuing task, step should return the new desired_goal.

    Regression for Issue #258: with continuing_task=True and reset_target=True, update_goal
    changes the internal goal after success, but the returned observation currently still
    contains the old desired_goal. The next policy call must see the updated goal.
    """
    maze_map = [
        [1, 1, 1, 1, 1],
        [1, "r", "g", "g", 1],
        [1, 1, 1, 1, 1],
    ]
    # position_noise_range cannot be passed through gym.make: PointMazeEnv forwards
    # unused kwargs into PointEnv/MujocoEnv, which rejects this MazeEnv-only argument.
    env = gym.make(
        "PointMaze_UMaze-v3",
        maze_map=maze_map,
        continuing_task=True,
        reset_target=True,
        reward_type="sparse",
        max_episode_steps=100,
    )
    try:
        env.unwrapped.position_noise_range = 0.0
        obs, _ = env.reset(
            seed=0,
            options={
                "reset_cell": np.array([1, 1]),
                "goal_cell": np.array([1, 2]),
            },
        )
        old_goal = obs["desired_goal"].copy()

        # Public API cannot teleport onto the goal without an expert policy or long
        # random rollouts; set MuJoCo state so the next step reliably triggers success.
        point_env = env.unwrapped.point_env
        point_env.set_state(
            np.concatenate([old_goal, point_env.data.qpos[2:]]),
            np.zeros_like(point_env.data.qvel),
        )

        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        new_goal = env.unwrapped.goal.copy()
        returned_desired_goal = obs["desired_goal"]

        assert info["success"] is True
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
        assert not np.allclose(old_goal, new_goal), (
            "Expected update_goal to switch the internal goal after success, "
            f"but old_goal={old_goal} new_goal={new_goal}"
        )
        np.testing.assert_allclose(
            returned_desired_goal,
            new_goal,
            err_msg=(
                "Returned desired_goal should match the post-step internal goal so the "
                "next policy call tracks the new target. "
                f"old_goal={old_goal}, returned_desired_goal={returned_desired_goal}, "
                f"internal_goal={new_goal}"
            ),
        )
        assert not np.allclose(returned_desired_goal, old_goal), (
            "Returned desired_goal should not remain the pre-switch goal. "
            f"old_goal={old_goal}, returned_desired_goal={returned_desired_goal}, "
            f"internal_goal={new_goal}"
        )
    finally:
        env.close()
