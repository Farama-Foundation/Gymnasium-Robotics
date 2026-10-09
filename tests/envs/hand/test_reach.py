import pickle

import gymnasium as gym
import mujoco
import numpy as np

import gymnasium_robotics
from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos

gym.register_envs(gymnasium_robotics)


def test_serialize_deserialize():
    env1 = gym.make("HandReach-v3", distance_threshold=1e-6)
    env1.reset()
    env2 = pickle.loads(pickle.dumps(env1))

    assert env1.unwrapped.distance_threshold == env2.unwrapped.distance_threshold, (
        env1.distance_threshold,
        env2.distance_threshold,
    )


# As in the mujoco-py implementation, relative control centers each finger's J1
# actuator on the sum of its J1 and J0 joint positions.
COUPLED_FINGER_JOINTS = {
    f"robot0:{finger}J1": f"robot0:{finger}J0" for finger in ("FF", "MF", "RF", "LF")
}


def test_relative_control_zero_action_matches_mujoco_py_targets():
    env = gym.make("HandReach-v3", relative_control=True)
    env.reset(seed=0)
    model, data = env.unwrapped.model, env.unwrapped.data
    targets = []
    for actuator_id in range(model.nu):
        actuator = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint = actuator.replace(":A_", ":")
        target = get_joint_qpos(model, data, joint)[0]
        if joint in COUPLED_FINGER_JOINTS:
            target += get_joint_qpos(model, data, COUPLED_FINGER_JOINTS[joint])[0]
        targets.append(target)
    targets = np.clip(
        targets, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    )

    env.step(np.zeros(env.action_space.shape))

    np.testing.assert_allclose(data.ctrl, targets)
    env.close()
