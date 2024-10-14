import pickle

import gymnasium as gym
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

ENVIRONMENT_IDS = (
    "HandManipulateEgg_ContinuousTouchSensors-v1",
    "HandManipulatePen_BooleanTouchSensors-v1",
    "HandManipulateBlock_BooleanTouchSensors-v1",
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
