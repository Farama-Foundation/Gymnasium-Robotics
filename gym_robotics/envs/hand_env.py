import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from gym_robotics.envs import robot_env


class HandEnv(robot_env.RobotEnv):
    def __init__(self, model_path, n_substeps, initial_qpos, relative_control, mujoco_bindings):
        self.relative_control = relative_control

        super().__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=20,
            initial_qpos=initial_qpos,
            mujoco_bindings=mujoco_bindings,
        )

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (20,)

        if self._mujoco_bindings.__name__ == "mujoco_py":
            ctrlrange = self.sim.model.actuator_ctrlrange
            actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
            if self.relative_control:
                actuation_center = np.zeros_like(action)
                for i in range(self.sim.data.ctrl.shape[0]):
                    actuation_center[i] = self.sim.data.get_joint_qpos(
                        self.sim.model.actuator_names[i].replace(":A_", ":")
                    )
                for joint_name in ["FF", "MF", "RF", "LF"]:
                    act_idx = self.sim.model.actuator_name2id(f"robot0:A_{joint_name}J1")
                    actuation_center[act_idx] += self.sim.data.get_joint_qpos(
                        f"robot0:{joint_name}J0"
                    )
            else:
                actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
            self.sim.data.ctrl[:] = actuation_center + action * actuation_range
            self.sim.data.ctrl[:] = np.clip(
                self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1]
            )
        else:
            ctrlrange = self.model.actuator_ctrlrange
            actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
            if self.relative_control:
                actuation_center = np.zeros_like(action)
                for i in range(self.data.ctrl.shape[0]):
                    actuation_center[i] = self.data.get_joint_qpos(
                        self.model.actuator_names[i].replace(":A_", ":")
                    )
                for joint_name in ["FF", "MF", "RF", "LF"]:
                    act_idx = self.model.actuator_name2id(f"robot0:A_{joint_name}J1")
                    actuation_center[act_idx] += self.data.get_joint_qpos(
                        f"robot0:{joint_name}J0"
                    )
            else:
                actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
            self.data.ctrl[:] = actuation_center + action * actuation_range
            self.data.ctrl[:] = np.clip(
                self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1]
            )


    def _viewer_setup(self):
        if self._mujoco_bindings.__name__ == "mujoco_py":  
            body_id = self.sim.model.body_name2id("robot0:palm")
            lookat = self.sim.data.body_xpos[body_id]
        else:
            body_id = self._mujoco_bindings.mj_name2id(self.model, self._mujoco_bindings.mjtObj.mjOBJ_BODY, "robot0:palm")
            lookat = self.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 55.0
        self.viewer.cam.elevation = -25.0

    def render(self, mode="human", width=500, height=500):
        return super().render(mode, width, height)
