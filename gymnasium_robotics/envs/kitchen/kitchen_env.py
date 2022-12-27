from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
from os import path
import numpy as np
from gymnasium_robotics.utils.rotations import euler2quat
from gymnasium_robotics.envs.kitchen.controller import IKController
import mujoco

MAX_CARTESIAN_DISPLACEMENT = 0.4
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
    def __init__(self, 
                    model_path="../assets/kitchen_franka/franka_assets/franka_panda.xml", 
                    frame_skip=50, 
                    observation_space=Box(low=-np.inf, high=np.inf, shape=(9, ), dtype=np.float32),
                    **kwargs):
        
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )
        
        self.control_step = 10

        super().__init__(xml_file_path, frame_skip, observation_space, **kwargs)
        
        # For the microwave kettle slide hinge
        self.init_qpos = np.array([ 1.48388023e-01, -1.76848573e+00,  1.84390296e+00, -2.47685760e+00,
                                    2.60252026e-01,  7.12533105e-01,  1.59515394e+00,  4.79267505e-02,
                                    3.71350919e-02, -2.66279850e-04, -5.18043486e-05,  3.12877220e-05,
                                   -4.51199853e-05, -3.90842156e-06, -4.22629655e-05,  6.28065475e-05,
                                    4.04984708e-05,  4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
                                   -6.44129196e-03, -1.77048263e-03,  1.08009684e-03, -2.69397440e-01,
                                    3.50383255e-01,  1.61944683e+00,  1.00618764e+00,  4.06395120e-03,
                                   -6.62095997e-03, -2.68278933e-04])
        
        self.controller = IKController(self.model, self.data)

        self._model_names = MujocoModelNames(self.model)

    def _get_obs(self):
        # q_pos, q_vel = robot_get_obs(self.model, self.data, self._model_names.joint_names)

        # Add robot noise

        return None

    def step(self, action):
            
        current_eef_pose = self.data.site_xpos[self._model_names.site_name2id['EEF']].copy()
        target_eef_pose = current_eef_pose + action[:3] * MAX_CARTESIAN_DISPLACEMENT
        quat_rot = euler2quat(action[3:6] * MAX_ROTATION_DISPLACEMENT)
        current_eef_quat = np.empty(4) # current orientation of the end effector site in quaternions
        target_orientation = np.empty(4)  # desired end effector orientation in quaternions
        mujoco.mju_mat2Quat(current_eef_quat, self.data.site_xmat[self._model_names.site_name2id['EEF']].copy())
        mujoco.mju_mulQuat(target_orientation, quat_rot, current_eef_quat)
        for _ in range(self.control_step):
            delta_qpos = self.controller.compute_qpos(current_eef_pose, current_eef_quat)
            ctrl_action = np.zeros(8)
            # print('DELTA QPOS')
            # print(delta_qpos)
            # ctrl_action[:7] = self.data.ctrl.copy()[:7] + delta_qpos[:7]
            # print(ctrl_action)
            self.do_simulation(ctrl_action, self.frame_skip)

            if self.render_mode == "human":
                self.render()

        observation = self._get_obs()
        
        return observation, 0.0, False, False, {}

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        return np.concatenate((position, velocity))

    def reset_model(self):

        qpos = self.init_qpos
        qvel = self.init_qvel
            
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation


class KitchenEnv(GoalEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        self.robot_env = FrankaRobot(model_path="../assets/kitchen_franka/kitchen_env_model.xml", **kwargs)
    def step(self, action):
        obs, _, _, _, _ = self.robot_env.step(action)
        rew = 0.0
        terminated = False
        truncated = False        
        
        return obs, rew, terminated, truncated, {}

    def reset(self):
        return None, 0.0, False, False, {}

    def render(self):
        self.robot_env.render()

