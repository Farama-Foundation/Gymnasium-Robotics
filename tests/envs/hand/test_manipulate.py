import pickle

import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics
from gymnasium_robotics.utils import rotations

gym.register_envs(gymnasium_robotics)
ENVIRONMENT_IDS = (
    #  "HandManipulateEgg-v0",
    #  "HandManipulatePen-v0",
    #  "HandManipulateBlock-v0",
    "HandManipulateEgg-v1",
    "HandManipulatePen-v1",
    "HandManipulateBlock-v1",
)


@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_serialize_deserialize(environment_id):
    env1 = gym.make(environment_id, target_position="fixed")
    env1.reset()
    env2 = pickle.loads(pickle.dumps(env1))

    assert env1.unwrapped.target_position == env2.unwrapped.target_position, (
        env1.target_position,
        env2.target_position,
    )


@pytest.mark.parametrize("batch_shape", [(2, 3), (4, 4)])
@pytest.mark.parametrize("reward_type", ["sparse", "dense"])
def test_reward_matches_individual_goals_for_multidimensional_batches(
    batch_shape, reward_type
):
    """Reward recomputation preserves independent goal pairs in nested batches."""
    with gym.make("HandManipulateBlock-v1", reward_type=reward_type) as env:
        obs, _ = env.reset(seed=0)
        achieved = np.broadcast_to(obs["achieved_goal"], batch_shape + (7,)).copy()
        desired = achieved.copy()
        rng = np.random.default_rng(0)
        desired[..., 3:] = rotations.euler2quat(
            rng.uniform(-np.pi, np.pi, batch_shape + (3,))
        )
        # Include successful transitions as well as nonzero rotation errors.
        desired[0, 0] = achieved[0, 0]
        expected = np.empty(batch_shape)
        for index in np.ndindex(batch_shape):
            expected[index] = env.unwrapped.compute_reward(
                achieved[index], desired[index], {}
            )

        result = env.unwrapped.compute_reward(achieved, desired, {})

        assert result.shape == batch_shape
        np.testing.assert_allclose(result, expected, atol=1e-12)
