"""Walker2d's factored observations follow the single-agent velocity bounds."""

import numpy as np
import pytest

from gymnasium_robotics import mamujoco_v1


@pytest.mark.parametrize("depth", [0, 1, 2, None])
@pytest.mark.parametrize("direction", ["state_to_local", "local_to_state"])
def test_walker_mapping_after_step(depth, direction):
    """A real legal step must preserve both public observation mapping directions."""
    env = mamujoco_v1.parallel_env("Walker2d", "2x3", agent_obsk=depth)
    try:
        env.reset(seed=123)
        action = np.random.default_rng(456).uniform(-1, 1, 6)
        observations, _, _, _, _ = env.step(
            env.map_global_action_to_local_actions(action)
        )
        state = env.state()
        if depth is not None:
            indices = np.concatenate(list(env.observation_factorization.values()))
            np.testing.assert_array_equal(np.unique(indices), np.arange(state.size))
        physics_velocities = env.single_agent_env.unwrapped.data.qvel.copy()
        assert np.any(np.abs(physics_velocities) > 10)
        env._get_obs()
        np.testing.assert_array_equal(
            env.single_agent_env.unwrapped.data.qvel, physics_velocities
        )
        if direction == "state_to_local":
            mapped = env.map_global_state_to_local_observations(state)
            for agent in env.possible_agents:
                np.testing.assert_allclose(mapped[agent], observations[agent])
        else:
            mapped = env.map_local_observations_to_global_state(observations)
            np.testing.assert_allclose(mapped, state)
    finally:
        env.close()


def test_custom_walker_keeps_its_observation_policy():
    """An explicit custom environment remains responsible for its velocity observation policy."""
    from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv

    class RawVelocityWalker(Walker2dEnv):
        def _get_obs(self):
            return np.concatenate((self.data.qpos[1:].copy(), self.data.qvel.copy()))

    env = mamujoco_v1.parallel_env("Walker2d", "2x3", gym_env=RawVelocityWalker())
    try:
        env.reset(seed=123)
        action = np.random.default_rng(456).uniform(-1, 1, 6)
        observations, _, _, _, _ = env.step(
            env.map_global_action_to_local_actions(action)
        )
        state = env.state()
        assert np.any(np.abs(env.single_agent_env.unwrapped.data.qvel) > 10)
        mapped = env.map_global_state_to_local_observations(state)
        for agent in env.possible_agents:
            np.testing.assert_allclose(mapped[agent], observations[agent])
        np.testing.assert_allclose(
            env.map_local_observations_to_global_state(observations), state
        )
    finally:
        env.close()
