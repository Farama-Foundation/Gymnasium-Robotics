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

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from gymnasium_robotics.envs.franka_kitchen.utils import (
    get_config_root_node,
    read_config_from_node,
)
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs

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
        "render_fps": 12,
    }

    def __init__(
        self,
        model_path="../assets/kitchen_franka/franka_assets/franka_panda.xml",
        frame_skip=40,
        robot_noise_ratio: float = 0.01,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

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

        self.init_qpos = self.data.qpos
        self.init_qvel = self.data.qvel

        self.act_mid = np.zeros(9)
        self.act_rng = np.ones(9) * 2
        config_path = path.join(
            path.dirname(__file__),
            "../assets/kitchen_franka/franka_assets/franka_config.xml",
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float64)
        self._read_specs_from_config(config_path)
        self.model_names = MujocoModelNames(self.model)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Denormalize the input action from [-1, 1] range to the each actuators control range
        action = self.act_mid + action * self.act_rng

        # enforce velocity limits
        ctrl_feasible = self._ctrl_velocity_limits(action)
        # enforce position limits
        ctrl_feasible = self._ctrl_position_limits(ctrl_feasible)

        self.do_simulation(ctrl_feasible, self.frame_skip)

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
        robot_qpos += (
            self.robot_noise_ratio
            * self.robot_pos_noise_amp[:9]
            * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qpos.shape)
        )
        robot_qvel += (
            self.robot_noise_ratio
            * self.robot_vel_noise_amp[:9]
            * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qvel.shape)
        )

        self._last_robot_qpos = robot_qpos

        return np.concatenate((robot_qpos.copy(), robot_qvel.copy()))

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        obs = self._get_obs()

        return obs

    def _ctrl_velocity_limits(self, ctrl_velocity: np.ndarray):
        """Enforce velocity limits and estimate joint position control input (to achieve the desired joint velocity).

        ALERT: This depends on previous observation. This is not ideal as it breaks MDP assumptions. This is the original
        implementation from the D4RL environment: https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/kitchen/adept_envs/franka/robot/franka_robot.py#L259.

        Args:
            ctrl_velocity (np.ndarray): environment action with space: Box(low=-1.0, high=1.0, shape=(9,))

        Returns:
            ctrl_position (np.ndarray): input joint position given to the MuJoCo simulation actuators.
        """
        ctrl_feasible_vel = np.clip(
            ctrl_velocity, self.robot_vel_bound[:9, 0], self.robot_vel_bound[:9, 1]
        )
        ctrl_feasible_position = self._last_robot_qpos + ctrl_feasible_vel * self.dt
        return ctrl_feasible_position

    def _ctrl_position_limits(self, ctrl_position: np.ndarray):
        """Enforce joint position limits.

        Args:
            ctrl_position (np.ndarray): unbounded joint position control input .

        Returns:
            ctrl_feasible_position (np.ndarray): clipped joint position control input.
        """
        ctrl_feasible_position = np.clip(
            ctrl_position, self.robot_pos_bound[:9, 0], self.robot_pos_bound[:9, 1]
        )
        return ctrl_feasible_position

    def _read_specs_from_config(self, robot_configs: str):
        """Read the specs of the Franka robot joints from the config xml file.
            - pos_bound: position limits of each joint.
            - vel_bound: velocity limits of each joint.
            - pos_noise_amp: scaling factor of the random noise applied in each observation of the robot joint positions.
            - vel_noise_amp: scaling factor of the random noise applied in each observation of the robot joint velocities.

        Args:
            robot_configs (str): path to 'franka_config.xml'
        """
        root, root_name = get_config_root_node(config_file_name=robot_configs)
        self.robot_name = root_name[0]
        self.robot_pos_bound = np.zeros([self.model.nv, 2], dtype=float)
        self.robot_vel_bound = np.zeros([self.model.nv, 2], dtype=float)
        self.robot_pos_noise_amp = np.zeros(self.model.nv, dtype=float)
        self.robot_vel_noise_amp = np.zeros(self.model.nv, dtype=float)

        for i in range(self.model.nv):
            self.robot_pos_bound[i] = read_config_from_node(
                root, "qpos" + str(i), "pos_bound", float
            )
            self.robot_vel_bound[i] = read_config_from_node(
                root, "qpos" + str(i), "vel_bound", float
            )
            self.robot_pos_noise_amp[i] = read_config_from_node(
                root, "qpos" + str(i), "pos_noise_amp", float
            )[0]
            self.robot_vel_noise_amp[i] = read_config_from_node(
                root, "qpos" + str(i), "vel_noise_amp", float
            )[0]
