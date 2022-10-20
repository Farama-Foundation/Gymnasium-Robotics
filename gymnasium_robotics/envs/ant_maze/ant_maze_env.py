import sys
from os import path
from typing import Dict, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v4 import AntEnv

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.ant_maze.maps import HARDEST_MAZE_EVAL
from gymnasium_robotics.envs.point_maze.maze import Maze
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


class AntMazeEnv(GoalEnv):
    """Ant navigating a maze."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        reward_type="dense",
        continuing_task=True,
        maze_map=HARDEST_MAZE_EVAL,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        # Get the ant.xml path from the Gymnasium package
        ant_xml_file_path = path.join(
            path.dirname(sys.modules[AntEnv.__module__].__file__), "assets/ant.xml"
        )
        # Add the maze to the xml
        self.maze, tmp_xml_file_path = Maze.make_maze(
            ant_xml_file_path, maze_map, maze_size_scaling=4, maze_height=0.5
        )

        # Create the MuJoCo environment, include position observation of the Ant for GoalEnv
        self.ant = AntEnv(
            xml_file=tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.ant.model)

        self.reward_type = reward_type
        self.continuing_task = continuing_task

        self.action_space = self.ant.action_space
        obs_shape: tuple = self.ant.observation_space.shape

        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape[0] - 2,), dtype="float64"
                ),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float64"),
            )
        )

        self.render_mode = render_mode
        self._render_initialized = True

        super().__init__(**kwargs)

    def generate_target_goal(self):
        assert len(self.maze.unique_goal_locations) > 0
        goal_index = self.np_random.integers(
            low=0, high=len(self.maze.unique_goal_locations)
        )
        goal = self.maze.unique_goal_locations[goal_index].copy()

        return goal

    def generate_reset_pos(self):
        assert len(self.maze.unique_reset_locations) > 0, ""

        # While reset position is close to goal position
        reset_pos = self.goal.copy()
        while np.linalg.norm(reset_pos - self.goal) <= 0.5:
            reset_index = self.np_random.integers(
                low=0, high=len(self.maze.unique_reset_locations)
            )
            reset_pos = self.maze.unique_reset_locations[reset_index].copy()

        return reset_pos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Dict[str, Optional[np.ndarray]] = {
            "goal_cell": None,
            "reset_cell": None,
        },
    ):
        super().reset(seed=seed)

        if options["goal_cell"] is not None:
            # assert that goal cell is valid
            assert self.maze.map_length > options["goal_cell"][1]
            assert self.maze.map_width > options["goal_cell"][0]
            assert (
                self.maze.maze_map[options["goal_cell"][1], options["goal_cell"][0]]
                != 1
            ), f"Goal can't be placed in a wall cell, {options['goal_cell']}"

            goal = self.maze.cell_rowcol_to_xy(options["goal_cell"])

        else:
            goal = self.generate_target_goal()

        # Add noise to goal position
        self.goal = self.add_xy_position_noise(goal)

        # Update target site for visualization
        site_id = self._model_names.site_name2id["target"]
        self.ant.model.site_pos[site_id] = np.append(
            self.goal, self.maze.maze_height / 2 * self.maze.maze_size_scaling
        )

        if options["reset_cell"] is not None:
            # assert that goal cell is valid
            assert self.maze.map_length > options["reset_cell"][1]
            assert self.maze.map_width > options["reset_cell"][0]
            assert (
                self.maze.maze_map[options["reset_cell"][1], options["reset_cell"][0]]
                != 1
            ), f"Reset can't be placed in a wall cell, {options['reset_cell']}"

            reset_pos = self.maze.cell_rowcol_to_xy(options["reset_cell"])

        else:
            reset_pos = self.generate_reset_pos()

        # Add noise to reset position
        reset_pos = self.add_xy_position_noise(reset_pos)

        self.ant.init_qpos[:2] = reset_pos

        ant_obs, ant_info = self.ant.reset(seed=seed)
        obs_dict = self._get_obs(ant_obs)

        return obs_dict, ant_info

    def _get_obs(self, ant_obs):
        achieved_goal = ant_obs[:2]
        observation = ant_obs[2:]

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == "dense":
            return -np.linalg.norm(desired_goal - achieved_goal)
        elif self.reward_type == "sparse":
            return 1.0 if np.linalg.norm(achieved_goal - desired_goal) <= 0.5 else 0.0

    def compute_terminated(self, achieved_goal, desired_goal, info):
        if not self.continuing_task:
            # If task is episodic terminate the episode when the goal is reached
            return bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.5)
        else:
            # Continuing tasks don't terminate, episode will be truncated when time limit is reached (`max_episode_steps`)
            return False

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return False

    def step(self, action):
        ant_obs, _, _, _, info = self.ant.step(action)
        obs = self._get_obs(ant_obs)

        terminated = self.compute_terminated(obs["achieved_goal"], self.goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        self.ant.render()
        # Set the viewer parameters at initialization of the renderer
        if self._render_initialized:
            self.ant.viewer.cam.distance = self.ant.model.stat.extent * 0.5
            self._render_initialized = False

    def close(self):
        super().close()
        self.ant.close()

    def add_xy_position_noise(self, xy_pos: np.ndarray) -> np.ndarray:
        noise_x = (
            self.np_random.uniform(low=-0.25, high=0.25) * self.maze.maze_size_scaling
        )
        noise_y = (
            self.np_random.uniform(low=-0.25, high=0.25) * self.maze.maze_size_scaling
        )
        xy_pos[0] += noise_x
        xy_pos[1] += noise_y

        return xy_pos
