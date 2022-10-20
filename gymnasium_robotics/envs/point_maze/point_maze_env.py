from os import path
from typing import Optional

import numpy as np
from gymnasium import spaces

# from gymnasium_robotics.envs.point_maze.point_env import PointEnv
from gymnasium_robotics.envs.ant_maze.maps import U_MAZE
from gymnasium_robotics.envs.point_maze.maze_env import MazeEnv
from gymnasium_robotics.envs.point_maze.point_env import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


class PointMazeEnv(MazeEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        point_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "../assets/point/point.xml"
        )
        super().__init__(
            agent_xml_path=point_xml_file_path,
            maze_map=U_MAZE,
            maze_size_scaling=1,
            maze_height=0.4,
            **kwargs,
        )

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path, render_mode=render_mode, **kwargs
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs_shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().reset(seed=seed, **kwargs)
        self.point_env.init_qpos[:2] = self.reset_pos

        obs, info = self.point_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)

        return obs_dict, info

    def step(self, action):
        obs, _, _, _, info = self.point_env.step(action)
        obs_dict = self._get_obs(obs)

        terminated = self.compute_terminated(obs_dict["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs_dict["achieved_goal"], self.goal, info)

        reward = self.compute_reward(obs_dict["achieved_goal"], self.goal, info)

        return obs_dict, reward, terminated, truncated, info

    def update_target_site_pos(self):
        self.point_env.model.site_pos[self.target_site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

    def _get_obs(self, point_obs):
        achieved_goal = point_obs[2:]
        return {
            "observation": point_obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def render(self):
        self.point_env.render()

    def close(self):
        super().close()
        self.point_env.close()
