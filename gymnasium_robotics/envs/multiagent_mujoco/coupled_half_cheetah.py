import os

import gymnasium
import numpy
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env


class CoupledHalfCheetah(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, agent_conf, render_mode: str = None):
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 20,
        }
        observation_space = gymnasium.spaces.Box(
            low=-numpy.inf, high=numpy.inf, shape=(35,), dtype=numpy.float32
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
            render_mode=render_mode,
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore1 = self.unwrapped.data.qpos[0]
        xposbefore2 = self.unwrapped.data.qpos[len(self.unwrapped.data.qpos) // 2]
        self.do_simulation(action, self.frame_skip)
        xposafter1 = self.unwrapped.data.qpos[0]
        xposafter2 = self.unwrapped.data.qpos[len(self.unwrapped.data.qpos) // 2]
        ob = self._get_obs()
        reward_ctrl1 = -0.1 * np.square(action[0 : len(action) // 2]).sum()
        reward_ctrl2 = -0.1 * np.square(action[len(action) // 2 :]).sum()
        reward_run1 = (xposafter1 - xposbefore1) / self.dt
        reward_run2 = (xposafter2 - xposbefore2) / self.dt
        reward = (reward_ctrl1 + reward_ctrl2) / 2.0 + (reward_run1 + reward_run2) / 2.0
        terminal = False
        return (
            ob,
            reward,
            terminal,
            False,
            dict(
                reward_run1=reward_run1,
                reward_ctrl1=reward_ctrl1,
                reward_run2=reward_run2,
                reward_ctrl2=reward_ctrl2,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.unwrapped.data.qpos.flat[1:],
                self.unwrapped.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.random(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_env_info(self):
        return {"episode_limit": self.episode_limit}
