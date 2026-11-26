from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


@pytest.mark.parametrize("version", ["v4", "v5", "v6"])
def test_reset(version):
    """Check that AntMaze does not reset into a success state."""
    env = gym.make(f"AntMaze_UMaze-{version}", continuing_task=True)

    for _ in range(1000):
        obs, info = env.reset()
        assert not info["success"]
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        assert dist > 0.45, f"dist={dist} < 0.45"


@pytest.mark.parametrize("version", ["v4", "v5", "v6"])
def test_temp_xml_file_lifecycle(version):
    """Check that the generated maze XML file is cleaned up when the environment closes."""
    env = gym.make(f"AntMaze_UMaze-{version}")

    try:
        unwrapped_env = env.unwrapped
        temp_xml_path = Path(unwrapped_env.tmp_xml_file_path)
        temp_dir = temp_xml_path.parent

        assert temp_xml_path.exists()
        assert temp_dir.exists()
    finally:
        env.close()

    assert not temp_xml_path.exists()
    assert not temp_dir.exists()


def _step_with_ant_torso_at_height(env, z):
    """Place the Ant's torso at height `z` (e.g. standing on a wall) and take one zero-action step."""
    unwrapped = env.unwrapped
    unwrapped.data.qpos[2] = z
    mujoco.mj_forward(unwrapped.model, unwrapped.data)
    return env.step(np.zeros(env.action_space.shape))


@pytest.mark.parametrize("continuing_task", [True, False])
def test_terminates_when_ant_is_unhealthy(continuing_task):
    """Check that AntMaze terminates when the Ant is unhealthy, e.g. on top of a wall (#259).

    The walls are 2 m tall and the Ant is healthy while its torso is within `healthy_z_range` = (0.2, 1.0).
    """
    env = gym.make("AntMaze_UMaze-v6", continuing_task=continuing_task)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, _ = _step_with_ant_torso_at_height(env, 2.5)

        assert terminated
        assert not truncated
    finally:
        env.close()


def test_does_not_terminate_when_ant_is_healthy():
    """Check that a healthy Ant that has not reached the goal does not terminate."""
    env = gym.make("AntMaze_UMaze-v6", continuing_task=True)
    try:
        env.reset(seed=0)
        _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape))

        assert not terminated
    finally:
        env.close()


def test_respects_terminate_when_unhealthy_false():
    """Check that `terminate_when_unhealthy=False` (a `gymnasium/MuJoCo/Ant` argument) is respected."""
    env = gym.make("AntMaze_UMaze-v6", terminate_when_unhealthy=False)
    try:
        env.reset(seed=0)
        _, _, terminated, _, _ = _step_with_ant_torso_at_height(env, 2.5)

        assert not terminated
    finally:
        env.close()


@pytest.mark.parametrize("continuing_task", [True, False])
def test_goal_termination_unchanged(continuing_task):
    """Check that reaching the goal terminates only when `continuing_task=False`."""
    env = gym.make("AntMaze_UMaze-v6", continuing_task=continuing_task)
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        unwrapped.data.qpos[:2] = unwrapped.goal
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))

        assert info["success"]
        assert terminated == (not continuing_task)
    finally:
        env.close()


@pytest.mark.parametrize("terminate_when_unhealthy", [False, True])
@pytest.mark.parametrize("continuing_task", [False, True])
def test_unhealthy_termination_can_be_recomputed(
    terminate_when_unhealthy, continuing_task
):
    """Recorded termination remains reproducible after the simulator resets."""
    with gym.make(
        "AntMaze_UMaze-v6",
        continuing_task=continuing_task,
        terminate_when_unhealthy=terminate_when_unhealthy,
    ) as env:
        env.reset(seed=0)
        obs, _, terminated, _, info = _step_with_ant_torso_at_height(env, 2.5)
        assert not info["success"]
        assert terminated == terminate_when_unhealthy
        for after_reset in (False, True):
            if after_reset:
                env.reset(seed=1)
            assert (
                env.unwrapped.compute_terminated(
                    obs["achieved_goal"], obs["desired_goal"], info
                )
                == terminated
            )


@pytest.mark.parametrize("info", [None, {}, [{"ant_terminated": True}]])
def test_compute_terminated_without_recorded_health_outcome(info):
    """`compute_terminated` falls back to the goal-based termination when `info` carries no Ant health outcome."""
    with gym.make("AntMaze_UMaze-v6", continuing_task=False) as env:
        env.reset(seed=0)
        goal = env.unwrapped.goal

        assert env.unwrapped.compute_terminated(goal, goal, info)
        assert not env.unwrapped.compute_terminated(goal + 5.0, goal, info)
