from os import path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.utils.rotations import quat2euler

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 45.0,
}


class AdroitHandHammerEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type: str = "dense", **kwargs):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "../assets/adroit_hand/adroit_hammer.xml",
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(46,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )
        self._model_names = MujocoModelNames(self.model)

        # whether to have sparse rewards
        if reward_type.lower() == "dense":
            self.sparse_reward = False
        elif reward_type.lower() == "sparse":
            self.sparse_reward = True
        else:
            raise ValueError(
                f"Unknown reward type, expected `dense` or `sparse` but got {reward_type}"
            )

        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )

        # change actuator sensitivity
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"]
            + 1,
            :3,
        ] = np.array([10, 0, 0])
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"]
            + 1,
            :3,
        ] = np.array([1, 0, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"]
            + 1,
            :3,
        ] = np.array([0, -10, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"]
            + 1,
            :3,
        ] = np.array([0, -1, 0])

        self.target_obj_site_id = self._model_names.site_name2id["S_target"]
        self.S_grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.obj_body_id = self._model_names.body_name2id["Object"]
        self.tool_site_id = self._model_names.site_name2id["tool"]
        self.goal_site_id = self._model_names.site_name2id["nail_goal"]
        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )

        EzPickle.__init__(self, **kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mean + a * self.act_rng  # mean center and scale

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        hamm_pos = self.data.xpos[self.obj_body_id].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_site_id].ravel()
        head_pos = self.data.site_xpos[self.tool_site_id].ravel()
        nail_pos = self.data.site_xpos[self.target_obj_site_id].ravel()
        goal_pos = self.data.site_xpos[self.goal_site_id].ravel()

        reward = 0.0
        if not self.sparse_reward:
            # get the palm to the hammer handle
            reward -= 0.1 * np.linalg.norm(palm_pos - hamm_pos)
            # take hammer head to nail
            reward -= np.linalg.norm(head_pos - nail_pos)
            # make nail go inside
            reward -= 10 * np.linalg.norm(nail_pos - goal_pos)
            # velocity penalty
            reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

        # bonus for lifting up the hammer
        if hamm_pos[2] > 0.04 and head_pos[2] > 0.04:
            reward += 2

        # bonus for hammering the nail
        if np.linalg.norm(nail_pos - goal_pos) < 0.020:
            reward += 25
        if np.linalg.norm(nail_pos - goal_pos) < 0.010:
            reward += 75

        goal_achieved = True if np.linalg.norm(nail_pos - goal_pos) < 0.010 else False

        if self.render_mode == "human":
            self.render()

        return obs, reward, False, False, dict(success=goal_achieved)

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
        obj_pos = self.data.xpos[self.obj_body_id].ravel()
        obj_rot = quat2euler(self.data.xquat[self.obj_body_id].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_site_id].ravel()
        target_pos = self.data.site_xpos[self.target_obj_site_id].ravel()
        nail_impact = np.clip(
            self.data.sensordata[self._model_names.sensor_name2id["S_nail"]], -1.0, 1.0
        )
        return np.concatenate(
            [
                qp[:-6],
                qv[-6:],
                palm_pos,
                obj_pos,
                obj_rot,
                target_pos,
                np.array([nail_impact]),
            ]
        )

    def reset_model(self):
        target_bid = self._model_names.body_name2id["nail_board"]
        self.model.body_pos[target_bid, 2] = self.np_random.uniform(low=0.1, high=0.25)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self.model.body_name2id("nail_board")].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)
