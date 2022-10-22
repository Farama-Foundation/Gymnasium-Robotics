import os

import numpy as np
from d4rl import offline_env
from d4rl.utils.quatmath import quat2euler
from gym import spaces, utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer

ADD_BONUS_REWARDS = True


class HammerEnvV0(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)
        self.target_obj_sid = -1
        self.S_grasp_sid = -1
        self.obj_bid = -1
        self.tool_sid = -1
        self.goal_sid = -1
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir + "/assets/DAPG_hammer.xml", 5)

        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )

        utils.EzPickle.__init__(self)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[
            self.sim.model.actuator_name2id("A_WRJ1") : self.sim.model.actuator_name2id(
                "A_WRJ0"
            )
            + 1,
            :3,
        ] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[
            self.sim.model.actuator_name2id("A_FFJ3") : self.sim.model.actuator_name2id(
                "A_THJ0"
            )
            + 1,
            :3,
        ] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[
            self.sim.model.actuator_name2id("A_WRJ1") : self.sim.model.actuator_name2id(
                "A_WRJ0"
            )
            + 1,
            :3,
        ] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[
            self.sim.model.actuator_name2id("A_FFJ3") : self.sim.model.actuator_name2id(
                "A_THJ0"
            )
            + 1,
            :3,
        ] = np.array([0, -1, 0])

        self.target_obj_sid = self.sim.model.site_name2id("S_target")
        self.S_grasp_sid = self.sim.model.site_name2id("S_grasp")
        self.obj_bid = self.sim.model.body_name2id("Object")
        self.tool_sid = self.sim.model.site_name2id("tool")
        self.goal_sid = self.sim.model.site_name2id("nail_goal")
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        # try:
        a = self.act_mid + a * self.act_rng  # mean center and scale
        # except:
        #     a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()

        # get to hammer
        reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)
        # take hammer head to nail
        reward -= np.linalg.norm(tool_pos - target_pos)
        # make nail go inside
        reward -= 10 * np.linalg.norm(target_pos - goal_pos)
        # velocity penalty
        reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

        if ADD_BONUS_REWARDS:
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 2

            # bonus for hammering the nail
            if np.linalg.norm(target_pos - goal_pos) < 0.020:
                reward += 25
            if np.linalg.norm(target_pos - goal_pos) < 0.010:
                reward += 75

        goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < 0.010 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_rot = quat2euler(self.data.body_xquat[self.obj_bid].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        nail_impact = np.clip(
            self.sim.data.sensordata[self.sim.model.sensor_name2id("S_nail")], -1.0, 1.0
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
        self.sim.reset()
        target_bid = self.model.body_name2id("nail_board")
        self.model.body_pos[target_bid, 2] = self.np_random.uniform(low=0.1, high=0.25)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self.model.body_name2id("nail_board")].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]
        board_pos = state_dict["board_pos"]
        self.set_state(qp, qv)
        self.model.body_pos[self.model.body_name2id("nail_board")] = board_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 45
        self.viewer.cam.distance = 2.0
        self.sim.forward()

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if nail insude board for 25 steps
        for path in paths:
            if np.sum(path["env_infos"]["goal_achieved"]) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
