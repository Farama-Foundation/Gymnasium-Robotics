from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.kitchen.controller import JointVelocityController
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
from os import path
import numpy as np


class FrankaRobot(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 500,
    }
    def __init__(self, 
                    model_path="../assets/kitchen_franka/franka_assets/franka_panda.xml", 
                    frame_skip=1, 
                    observation_space=Box(low=-np.inf, high=np.inf, shape=(9, ), dtype=np.float32),
                    **kwargs):
        
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        super().__init__(xml_file_path, frame_skip, observation_space, **kwargs)

        self._model_names = MujocoModelNames(self.model)

        self.simulation_timestep = self.model.opt.timestep
        self.control_timestep = 0.05 # Control frequency of 20 Hz

        # Joint control limits of the model
        torque_actuator_idx = []
        for name in self._model_names.actuator_names:
            if 'gripper' not in name:
                torque_actuator_idx.append(self._model_names.actuator_name2id[name])

        low = self.model.actuator_ctrlrange[torque_actuator_idx, 0]
        high = self.model.actuator_ctrlrange[torque_actuator_idx, 1]
        self.velocity_controller = JointVelocityController(actuator_range=[low, high], velocity_limits=[-1,1])

    def _get_obs(self):
        q_pos, q_vel = robot_get_obs(self.model, self.data, self._model_names.joint_names)

        # Add robot noise

        return q_pos, q_vel

    def step(self, action):

        action = np.clip(action, -1.0, 1.0)

        action = self.act_mid + action * self.act_amp  # mean center and scale

        policy_step = True
        for i in range(int(self.control_timestep/self.simulation_timestep)):
            if policy_step:
                self.velocity_controller.set_goal(action)
            torques = self.velocity_controller.run_controller()
            self.do_simulation(torques, self.frame_skip)
            policy_step = False

        observation = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return observation, 0.0, False, False, {}}

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        pass
    


# class KitchenEnv(GoalEnv, utils.EzPickle):
#     def __init__(self):


#         self.robot_env = FrankaRobot()
#         pass

#     def step(self, action):
#         pass

#     def reset(self):
#         pass

#     def render(self):
#         pass

