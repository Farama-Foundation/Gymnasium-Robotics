"""Environment using Gymnasium API for Franka robot.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

This code was also implemented over the repository relay-policy-learning on GitHub (https://github.com/google-research/relay-policy-learning),
published in Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning, by
Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman

Original Author of the code: Abhishek Gupta & Justin Fu

The modifications made involve separatin the Kitchen environment from the Franka environment and addint support for compatibility with
the Gymnasium and Multi-goal API's

This project is covered by the Apache 2.0 License.
"""

from os import path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from gymnasium_robotics.envs.franka_kitchen.ik_controller import IKController
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs
from gymnasium_robotics.utils.rotations import euler2quat

MAX_CARTESIAN_DISPLACEMENT = 0.2
MAX_ROTATION_DISPLACEMENT = 0.5

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 70.0,
    "elevation": -35.0,
    "lookat": np.array([-0.2, 0.5, 2.0]),
}


class FrankaRobot(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
        self,
        model_path="../assets/kitchen_franka/franka_assets/franka_panda.xml",
        frame_skip=50,
        ik_controller: bool = True,
        control_steps=5,
        robot_noise_ratio: float = 0.01,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        self.control_steps = control_steps
        self.robot_noise_ratio = robot_noise_ratio

        observation_space = (
            spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
        )

        super().__init__(
            xml_file_path,
            frame_skip,
            observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.init_ctrl = np.array([0, 0, 0, -1.57079, 0, 1.57079, 0, 255])

        if ik_controller:
            self.controller = IKController(self.model, self.data)
            action_size = 7  # 3 translation + 3 rotation + 1 gripper

        else:
            self.controller = None
            action_size = 8  # 7 joint positions + 1 gripper

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=(action_size,)
        )

        # Actuator ranges
        ctrlrange = self.model.actuator_ctrlrange
        self.actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        self.actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0

        self.model_names = MujocoModelNames(self.model)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        if self.controller is not None:
            current_eef_pose = self.data.site_xpos[
                self.model_names.site_name2id["EEF"]
            ].copy()
            target_eef_pose = current_eef_pose + action[:3] * MAX_CARTESIAN_DISPLACEMENT
            quat_rot = euler2quat(action[3:6] * MAX_ROTATION_DISPLACEMENT)
            current_eef_quat = np.empty(
                4
            )  # current orientation of the end effector site in quaternions
            target_orientation = np.empty(
                4
            )  # desired end effector orientation in quaternions
            mujoco.mju_mat2Quat(
                current_eef_quat,
                self.data.site_xmat[self.model_names.site_name2id["EEF"]].copy(),
            )
            mujoco.mju_mulQuat(target_orientation, quat_rot, current_eef_quat)

            ctrl_action = np.zeros(8)

            # Denormalize gripper action
            ctrl_action[-1] = (
                self.actuation_center[-1] + action[-1] * self.actuation_range[-1]
            )

            for _ in range(self.control_steps):
                delta_qpos = self.controller.compute_qpos_delta(
                    target_eef_pose, target_orientation
                )
                ctrl_action[:7] = self.data.ctrl.copy()[:7] + delta_qpos[:7]

                # Do not use `do_simulation`` method from MujocoEnv: value error due to discrepancy between
                # the action space and the simulation control input when using IK controller.
                # TODO: eliminate error check in MujocoEnv (action space can be different from simulaton control input).
                self.data.ctrl[:] = ctrl_action
                mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

                if self.render_mode == "human":
                    self.render()
        else:
            # Denormalize the input action from [-1, 1] range to the each actuators control range
            action = self.actuation_center + action * self.actuation_range
            self.do_simulation(action, self.frame_skip)
            if self.render_mode == "human":
                self.render()

        obs = self._get_obs()

        return obs, 0.0, False, False, {}

    def _get_obs(self):
        # Gather simulated observation
        robot_qpos, robot_qvel = robot_get_obs(
            self.model, self.data, self.model_names.joint_names
        )

        # Simulate observation noise
        robot_qpos += self.robot_noise_ratio * self.np_random.uniform(
            low=-1.0, high=1.0, size=robot_qpos.shape
        )
        robot_qvel += self.robot_noise_ratio * self.np_random.uniform(
            low=-1.0, high=1.0, size=robot_qvel.shape
        )

        return np.concatenate((robot_qpos.copy(), robot_qvel.copy()))

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.data.ctrl[:] = self.init_ctrl
        self.set_state(qpos, qvel)

        obs = self._get_obs()

        return obs
