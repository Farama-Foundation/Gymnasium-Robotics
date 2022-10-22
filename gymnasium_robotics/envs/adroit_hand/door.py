from os import path

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames

ADD_BONUS_REWARDS = True


class DoorEnv(MujocoEnv, EzPickle):
    def __init__(self, **kwargs):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "../assets/adroit/adroit_door.xml"
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )
        MujocoEnv().__init__(
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            **kwargs
        )
        self._utils = MujocoModelNames(self.model, self.data)

        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )
        # change actuator sensitivity
        self.model.actuator_gainprm[
            self._utils.actuator_name2id("A_WRJ1") : self._utils.actuator_name2id(
                "A_WRJ0"
            )
            + 1,
            :3,
        ] = np.array([10, 0, 0])
        self.model.actuator_gainprm[
            self._utils.actuator_name2id("A_FFJ3") : self._utils.actuator_name2id(
                "A_THJ0"
            )
            + 1,
            :3,
        ] = np.array([1, 0, 0])
        self.model.actuator_biasprm[
            self._utils.actuator_name2id("A_WRJ1") : self._utils.actuator_name2id(
                "A_WRJ0"
            )
            + 1,
            :3,
        ] = np.array([0, -10, 0])
        self.model.actuator_biasprm[
            self._utils.actuator_name2id("A_FFJ3") : self._utils.actuator_name2id(
                "A_THJ0"
            )
            + 1,
            :3,
        ] = np.array([0, -1, 0])

        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )
        self.door_hinge_addrs = self.model.jnt_dofadr[
            self._utils.joint_name2id("door_hinge")
        ]
        self.grasp_site_id = self._utils.site_name2id("S_grasp")
        self.handle_site_id = self._utils.site_name2id("S_handle")
        self.door_body_id = self._utils.body_name2id("frame")

        EzPickle.__init__(self, **kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        # try:
        a = self.act_mid + a * self.act_rng  # mean center and scale
        # except:
        #     a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        handle_pos = self.site_xpos[self.handle_site_id].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        # get to handle
        reward = -0.1 * np.linalg.norm(palm_pos - handle_pos)
        # open door
        reward += -0.1 * (door_pos - 1.57) * (door_pos - 1.57)
        # velocity cost
        reward += -1e-5 * np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARDS:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        goal_achieved = True if door_pos >= 1.35 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_site_id].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_addrs]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]
        return np.concatenate(
            [
                qp[1:-2],
                [latch_pos],
                door_pos,
                palm_pos,
                handle_pos,
                palm_pos - handle_pos,
                [door_open],
            ]
        )

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid, 0] = self.np_random.uniform(
            low=-0.3, high=-0.2
        )
        self.model.body_pos[self.door_bid, 1] = self.np_random.uniform(
            low=0.25, high=0.35
        )
        self.model.body_pos[self.door_bid, 2] = self.np_random.uniform(
            low=0.252, high=0.35
        )

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict["door_body_pos"]
        self.sim.forward()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.azimuth = 90
        self.viewer.cam.distance = 1.5
