import pickle

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import gymnasium_robotics

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


@pytest.mark.parametrize("target_rotation", ["fixed", "ignore"])
@pytest.mark.parametrize("environment_id", ENVIRONMENT_IDS)
def test_unrandomized_target_rotation_is_the_object_rotation(
    environment_id, target_rotation
):
    env = gym.make(environment_id, target_rotation=target_rotation)
    obs, _ = env.reset(seed=0)

    np.testing.assert_allclose(obs["desired_goal"][3:], obs["achieved_goal"][3:])
    env.close()


def test_initial_qpos_sets_the_reset_configuration():
    env = gym.make("HandManipulateBlock-v1", initial_qpos={"robot0:WRJ1": 0.1})
    model = env.unwrapped.model
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:WRJ1")

    assert env.unwrapped.initial_qpos[model.jnt_qposadr[joint_id]] == 0.1
    env.close()
