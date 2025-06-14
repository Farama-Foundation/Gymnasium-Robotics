import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

EPS = 1e-6


def verify_environments_match(
    old_env_id: str, new_env_id: str, seed: int = 1, num_actions: int = 1000
):
    """Verifies with two environment ids (old and new) are identical in obs, reward and done
    (except info where all old info must be contained in new info)."""
    old_env = gym.make(old_env_id)
    new_env = gym.make(new_env_id)

    old_reset_obs, old_info = old_env.reset(seed=seed)
    new_reset_obs, new_info = new_env.reset(seed=seed)

    np.testing.assert_allclose(old_reset_obs, new_reset_obs)

    for i in range(num_actions):
        action = old_env.action_space.sample()
        old_obs, old_reward, old_terminated, old_truncated, old_info = old_env.step(
            action
        )
        new_obs, new_reward, new_terminated, new_truncated, new_info = new_env.step(
            action
        )

        np.testing.assert_allclose(old_obs, new_obs, atol=EPS)
        np.testing.assert_allclose(old_reward, new_reward, atol=EPS)
        np.testing.assert_equal(old_terminated, new_terminated)
        np.testing.assert_equal(old_truncated, new_truncated)

        for key in old_info:
            np.testing.assert_allclose(old_info[key], new_info[key], atol=EPS)

        if old_terminated or old_truncated:
            break


MUJOCO_PY_ENVS = [
    "ant-v2",
    "ant-v3",
    "halfcheetah-v2",
    "halfcheetah-v3",
    "hopper-v2",
    "hopper-v3",
    "humanoid-v2",
    "humanoid-v3",
    "humanoid_standup-v2",
    "inverted_double_pendulum-v2",
    "inverted_pendulum-v2",
    "pusher-v2",
    "reacher-v2",
    "swimmer-v2",
    "swimmer-v3",
    "walker2d-v2",
    "walker2d-v3",
]

MUJOCO_V2_V3_ENVS = [
    "ant" "halfcheetah",
    "hopper",
    "humanoid",
    "swimmer",
    "walker2d",
]


@pytest.mark.parametrize("env_name", MUJOCO_V2_V3_ENVS)
def test_mujoco_v2_to_v3_conversion(env_name: str):
    """Checks that all v2 mujoco environments are the same as v3 environments."""
    verify_environments_match(f"{env_name}-v2", f"{env_name}-v3")


@pytest.mark.parametrize("env_name", MUJOCO_V2_V3_ENVS)
def test_mujoco_incompatible_v3_to_v2(env_name: str):
    """Checks that the v3 environment are slightly different from v2, (v3 has additional info keys that v2 does not)."""
    with pytest.raises(KeyError):
        verify_environments_match(f"{env_name}-v3", f"{env_name}-v2")
