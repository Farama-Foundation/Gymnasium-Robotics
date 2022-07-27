import os
from gym import utils
from gym_robotics.envs.fetch_env import MujocoFetchEnv, MujocoPyFetchEnv


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "reach.xml")


class MujocoPyFetchReachEnv(MujocoPyFetchEnv, utils.EzPickle):
    def __init__(self, mujoco_bindings, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        MujocoPyFetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            mujoco_bindings=mujoco_bindings,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)
