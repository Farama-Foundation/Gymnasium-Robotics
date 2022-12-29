from os import path

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.kitchen.controller import IKController
from gymnasium_robotics.utils.mujoco_utils import (
    MujocoModelNames,
    get_joint_qpos,
    get_joint_qvel,
    robot_get_obs,
)
from gymnasium_robotics.utils.rotations import euler2quat

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 70.0,
    "elevation": -35.0,
    "lookat": np.array([-0.2, 0.5, 2.0]),
}

OBS_ELEMENT_GOALS = {
    "bottom_right_burner": np.array([-0.01]),
    "bottom_left_burner": np.array([-0.01]),
    "top_right_burner": np.array([-0.01]),
    "top_left_burner": np.array([-0.01]),
    "light_switch": np.array([-0.7]),
    "slide_cabinet": np.array([0.37]),
    "left_hinge_cabinet": np.array([-1.45]),
    "right_hinge_cabinet": np.array([1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3
MAX_CARTESIAN_DISPLACEMENT = 0.2
MAX_ROTATION_DISPLACEMENT = 0.5


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
        observation_space=Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
        ik_controller: bool = True,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs
    ):

        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        self.control_step = 5

        super().__init__(
            xml_file_path,
            frame_skip,
            observation_space,
            default_camera_config=default_camera_config,
            **kwargs
        )

        self.init_ctrl = np.array([0, 0, 0, -1.57079, 0, 1.57079, 0, 255])
        if ik_controller:
            self.controller = IKController(self.model, self.data)
        else:
            self.controller = None

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
            for _ in range(self.control_step):
                delta_qpos = self.controller.compute_qpos(
                    target_eef_pose, target_orientation
                )
                ctrl_action[:7] = self.data.ctrl.copy()[:7] + delta_qpos[:7]
                self.do_simulation(ctrl_action, self.frame_skip)

                if self.render_mode == "human":
                    self.render()
        else:
            # Joit position control
            # un-normalize with max and min ctrl range
            self.data.ctrl[:7] = action
            pass
        obs = {}
        obs["time"], obs["robot_qpos"], obs["robot_qvel"] = self._get_obs()

        return obs, 0.0, False, False, {}

    def _get_obs(self):
        # Gather simulated observation
        obs = {}
        obs["robot_qpos"], obs["robot_qvel"] = robot_get_obs(
            self.model, self.data, self.model_names.joint_names
        )

        obs["t"] = self.data.time
        # Simulate observation noise
        # qp += robot_noise_ratio*self.robot_pos_noise_amp[:self.n_jnt]*env.np_random.uniform(low=-1., high=1., size=self.n_jnt)
        # qv += robot_noise_ratio*self.robot_vel_noise_amp[:self.n_jnt]*env.np_random.uniform(low=-1., high=1., size=self.n_jnt)

        return obs

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.data.ctrl[:] = self.init_ctrl
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation


class KitchenEnv(GoalEnv, EzPickle):
    def __init__(
        self,
        tasks_to_complete: "list[str]" = list(OBS_ELEMENT_GOALS.keys()),
        terminate_on_task_completed: bool = True,
        **kwargs
    ):
        self.robot_env = FrankaRobot(
            model_path="../assets/kitchen_franka/kitchen_env_model.xml", **kwargs
        )
        self.robot_env.init_qpos[:7] = np.array(
            [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.0]
        )
        self.model = self.robot_env.model
        self.data = self.robot_env.data

        self.terminate_on_task_completed = terminate_on_task_completed

        self.goal_qpos = np.concatenate(
            [OBS_ELEMENT_GOALS[task_qpos] for task_qpos in tasks_to_complete]
        )
        self.goal_tasks = set(tasks_to_complete).copy()
        self.tasks_to_complete = set(tasks_to_complete)

    def _get_reward_and_obs(self):
        obj_qpos = []
        obj_qvel = []
        completions = []
        for joint_name in self.robot_env.model_names.joint_names:
            if not joint_name.startswith("robot"):
                qpos = get_joint_qpos(self.model, self.data, joint_name)
                obj_qpos.append(qpos)
                obj_qvel.append(get_joint_qvel(self.model, self.data, joint_name))
                if joint_name in self.tasks_to_complete:
                    distance = np.linalg.norm(qpos - OBS_ELEMENT_GOALS[joint_name])
                    complete = distance < BONUS_THRESH
                    if complete:
                        completions.append(joint_name)

        # When the task is accomplished remove from tasks to be completed
        [self.tasks_to_complete.remove(element) for element in completions]
        reward = float(len(completions))

        # TODO: Add noise to object observations
        # qp_obj += robot_noise_ratio*self.robot_pos_noise_amp[-self.n_obj:]*env.np_random.uniform(low=-1., high=1., size=self.n_obj)
        # qv_obj += robot_noise_ratio*self.robot_vel_noise_amp[-self.n_obj:]*env.np_random.uniform(low=-1., high=1., size=self.n_obj)

        return reward, np.concatenate(obj_qpos).copy(), np.concatenate(obj_qvel).copy()

    def step(self, action):
        obs, _, terminated, truncated, info = self.robot_env.step(action)
        reward, obs["obj_qpos"], obs["obj_qvel"] = self._get_reward_and_obs()
        obs["goal"] = self.goal_qpos

        if self.terminate_on_task_completed:
            # terminate if there are no more tasks to complete
            terminated = not self.tasks_to_complete

        return obs, reward, terminated, truncated, info

    def reset(self):
        obs, info = self.robot_env.reset()
        self.task_to_complete = self.goal_tasks.copy()

        return obs, info

    def render(self):
        self.robot_env.render()
