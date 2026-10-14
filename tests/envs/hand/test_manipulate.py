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


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.parametrize("reward_type", ["sparse", "dense"])
def test_pen_batched_rewards_match_individual_rewards(batch_size, reward_type):
    """Ignoring pen rotation about z must work independently for each goal."""
    achieved_euler = np.array(
        [[0.2, 0.3, 0.5], [-0.3, 0.2, -0.7], [0.5, -0.4, 0.8], [0.1, 0.2, -0.2]]
    )[:batch_size]
    desired_euler = np.array(
        [[0.2, 0.3, -0.5], [0.3, 0.2, 0.7], [-0.5, 0.4, -0.8], [0.1, 0.2, 0.2]]
    )[:batch_size]
    achieved_goal = np.concatenate(
        [np.zeros((batch_size, 3)), rotations.euler2quat(achieved_euler)], axis=-1
    )
    desired_goal = np.concatenate(
        [np.zeros((batch_size, 3)), rotations.euler2quat(desired_euler)], axis=-1
    )
    achieved_before = achieved_goal.copy()
    desired_before = desired_goal.copy()

    with gym.make("HandManipulatePen-v1", reward_type=reward_type) as env:
        individual_rewards = np.array(
            [
                env.unwrapped.compute_reward(achieved, desired, {})
                for achieved, desired in zip(achieved_goal, desired_goal)
            ]
        )
        batched_rewards = env.unwrapped.compute_reward(achieved_goal, desired_goal, {})

    np.testing.assert_allclose(batched_rewards, individual_rewards, atol=1e-7)
    np.testing.assert_array_equal(achieved_goal, achieved_before)
    np.testing.assert_array_equal(desired_goal, desired_before)
