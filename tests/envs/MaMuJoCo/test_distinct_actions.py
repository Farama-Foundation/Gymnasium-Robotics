import gymnasium as gym
import numpy as np
import pytest

from gymnasium_robotics import mamujoco_v1


@pytest.mark.parametrize("endpoint", ["low", "high"])
def test_unequal_bounds_route_distinct_actions(endpoint):
    low = -np.arange(1, 9, dtype=np.float32)
    high = np.arange(2, 10, dtype=np.float32)
    base = gym.wrappers.RescaleAction(
        gym.make("Ant-v5"), min_action=low, max_action=high
    )
    env = mamujoco_v1.parallel_env("Ant", "2x4", gym_env=base)
    try:
        env.reset(seed=2)
        fractions = np.linspace(0.1, 0.8, 8, dtype=np.float32)
        if endpoint == "high":
            fractions = 1 - fractions
        controls = low + fractions * (high - low)
        indices = [[2, 3, 4, 5], [6, 7, 0, 1]]
        actions = {a: controls[i] for a, i in zip(env.possible_agents, indices)}
        for agent, action in actions.items():
            assert env.action_space(agent).contains(action)
        env.step(actions)
        np.testing.assert_allclose(
            base.unwrapped.data.ctrl, 2 * fractions - 1, atol=1e-6
        )
    finally:
        env.close()
