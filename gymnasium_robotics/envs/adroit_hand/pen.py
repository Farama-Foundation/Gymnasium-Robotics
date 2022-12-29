from os import path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.utils.rotations import euler2quat


class AdroitHandPenEnv(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, sparse_reward=False, **kwargs):
        self.pen_length = 1.0
        self.tar_length = 1.0

        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "../assets/adroit_hand/adroit_pen.xml",
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs
        )
        self._model_names = MujocoModelNames(self.model)

        # whether to have sparse rewards
        self.sparse_reward = sparse_reward

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

        self.target_obj_body_id = self._model_names.body_name2id["target"]
        self.S_grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.obj_body_id = self._model_names.body_name2id["Object"]
        self.eps_ball_site_id = self._model_names.site_name2id["eps_ball"]
        self.obj_t_site_id = self._model_names.site_name2id["object_top"]
        self.obj_b_site_id = self._model_names.site_name2id["object_bottom"]
        self.tar_t_site_id = self._model_names.site_name2id["target_top"]
        self.tar_b_site_id = self._model_names.site_name2id["target_bottom"]

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

        obj_pos = self.data.xpos[self.obj_body_id].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_site_id].ravel()
        obj_orien = (
            self.data.site_xpos[self.obj_t_site_id]
            - self.data.site_xpos[self.obj_b_site_id]
        ) / self.pen_length
        desired_orien = (
            self.data.site_xpos[self.tar_t_site_id]
            - self.data.site_xpos[self.tar_b_site_id]
        ) / self.tar_length

        # pos cost
        dist = np.linalg.norm(obj_pos - desired_loc)
        reward = -dist * (not self.sparse_reward)
        # orien cost
        orien_similarity = np.dot(obj_orien, desired_orien)
        reward += orien_similarity * (not self.sparse_reward)

        # bonus for being close to desired orientation
        if dist < 0.075 and orien_similarity > 0.9:
            reward += 10
        if dist < 0.075 and orien_similarity > 0.95:
            reward += 50

        # penalty for dropping the pen
        terminated = False
        if obj_pos[2] < 0.075:
            reward -= 5
            terminated = True

        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, dict(success=goal_achieved)

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.xpos[self.obj_body_id].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_site_id].ravel()
        obj_orien = (
            self.data.site_xpos[self.obj_t_site_id]
            - self.data.site_xpos[self.obj_b_site_id]
        ) / self.pen_length
        desired_orien = (
            self.data.site_xpos[self.tar_t_site_id]
            - self.data.site_xpos[self.tar_b_site_id]
        ) / self.tar_length

        return np.concatenate(
            [
                qpos[:-6],
                obj_pos,
                obj_vel,
                obj_orien,
                desired_orien,
                obj_pos - desired_pos,
                obj_orien - desired_orien,
            ]
        )

    def reset_model(self):
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_body_id] = euler2quat(desired_orien)

        self.set_state(self.init_qpos, self.init_qvel)

        self.pen_length = np.linalg.norm(
            self.data.site_xpos[self.obj_t_site_id]
            - self.data.site_xpos[self.obj_b_site_id]
        )
        self.tar_length = np.linalg.norm(
            self.data.site_xpos[self.tar_t_site_id]
            - self.data.site_xpos[self.tar_b_site_id]
        )

        obs = self._get_obs()
        return obs

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def viewer_setup(self):
        self.viewer.cam.azimuth = -45
        self.viewer.cam.distance = 1.0
