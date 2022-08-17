import numpy as np
from typing import Union
from gym_robotics.envs.robot_env import MujocoRobotEnv, MujocoPyRobotEnv


def get_base_hand_env(
    RobotEnvClass: Union[MujocoPyRobotEnv, MujocoRobotEnv]
) -> Union[MujocoPyRobotEnv, MujocoRobotEnv]:
    class BaseHandEnv(RobotEnvClass):
        def __init__(self, model_path, n_substeps, initial_qpos, relative_control):
            self.relative_control = relative_control

            super().__init__(
                model_path=model_path,
                n_substeps=n_substeps,
                n_actions=20,
                initial_qpos=initial_qpos,
            )

        # RobotEnv methods
        # ----------------------------
        def _get_palm_xpos(self):

            NotImplementedError

        def _set_action(self, action):
            assert action.shape == (20,)

        def _viewer_setup(self):
            lookat = self._get_palm_xpos()
            for idx, value in enumerate(lookat):
                self.viewer.cam.lookat[idx] = value
            self.viewer.cam.distance = 0.5
            self.viewer.cam.azimuth = 55.0
            self.viewer.cam.elevation = -25.0

        def render(self, mode="human", width=500, height=500):
            return super().render(mode, width, height)

    return BaseHandEnv


class MujocoHandEnv(get_base_hand_env(MujocoRobotEnv)):
    def __init__(self, model_path, n_substeps, initial_qpos, relative_control) -> None:
        super().__init__(model_path, n_substeps, initial_qpos, relative_control)

    def _set_action(self, action):
        super()._set_action(action)
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
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def _get_palm_xpos(self):
        body_id = self._model_names.body_name2id["robot0:palm"]
        return self.data.xpos[body_id]


class MujocoPyHandEnv(get_base_hand_env(MujocoPyRobotEnv)):
    def __init__(self, model_path, n_substeps, initial_qpos, relative_control) -> None:
        super().__init__(model_path, n_substeps, initial_qpos, relative_control)

    def _set_action(self, action):
        super()._set_action(action)

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

    def _get_palm_xpos(self):
        body_id = self.sim.model.body_name2id("robot0:palm")
        return self.sim.data.body_xpos[body_id]
