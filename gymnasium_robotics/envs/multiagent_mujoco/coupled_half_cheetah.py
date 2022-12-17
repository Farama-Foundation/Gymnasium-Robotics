import os

import gymnasium
import numpy as np
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.utils.ezpickle import EzPickle

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class CoupledHalfCheetah(mujoco_env.MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, render_mode: str = None):
        self._forward_reward_weight = 1
        self._ctrl_cost_weight = 0.1
        self._reset_noise_scale = 0.1

        observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32
        )

        mujoco_env.MujocoEnv.__init__(
            self,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "assets",
                "coupled_half_cheetah.xml",
            ),
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
        )
        EzPickle.__init__(self)

    def step(self, action):
        xposbefore1 = self.data.qpos[0]
        xposbefore2 = self.data.qpos[len(self.data.qpos) // 2]
        self.do_simulation(action, self.frame_skip)
        x_position_after1 = self.data.qpos[0]
        x_position_after2 = self.data.qpos[len(self.data.qpos) // 2]
        x_velocity1 = (x_position_after1 - xposbefore1) / self.dt  # velocity
        x_velocity2 = (x_position_after2 - xposbefore2) / self.dt  # velocity

        ctrl_cost1 = (
            self._ctrl_cost_weight * np.square(action[0 : len(action) // 2]).sum()
        )
        ctrl_cost2 = (
            self._ctrl_cost_weight * np.square(action[len(action) // 2 :]).sum()
        )

        forward_reward = self._forward_reward_weight * (x_velocity1 + x_velocity2) / 2.0

        observation = self._get_obs()
        reward = forward_reward - (ctrl_cost1 + ctrl_cost2) / 2.0
        terminal = False
        truncated = False
        info = {
            "x_position1": x_position_after1,
            "x_position2": x_position_after2,
            "x_velocity1": x_velocity1,
            "x_velocity2": x_velocity2,
            "reward_run": forward_reward,
            "reward_ctrl1": ctrl_cost1,
            "reward_ctrl2": ctrl_cost2,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminal, truncated, info

    def _get_obs(self):
        # NOTE: does not return tendon data
        return np.concatenate(
            [
                self.data.qpos.flat[1:],
                self.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        qvel = (
            self.init_qvel
            + self.np_random.random(self.model.nv) * self._reset_noise_scale
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
