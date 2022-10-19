import sys
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from os import path
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v4 import AntEnv

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.ant.maps import GOAL, HARDEST_MAZE_EVAL_TEST, RESET


def add_maze_to_xml(xml_path: str, maze_map: list, maze_size_scaling: int):

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    maze_height = 0.5
    maze_size_scaling = 4

    def find_robot():
        structure = maze_map
        size_scaling = maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == RESET:
                    return j * size_scaling, i * size_scaling
        raise ValueError("No robot in maze specification.")

    # Obtain a numpy array form for a maze map in case we want to reset
    # to multiple starting states
    temp_maze_map = deepcopy(maze_map)
    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            if temp_maze_map[i][j] in [
                RESET,
            ]:
                temp_maze_map[i][j] = 0
            elif temp_maze_map[i][j] in [
                GOAL,
            ]:
                temp_maze_map[i][j] = 1

    torso_x, torso_y = find_robot()

    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            struct = maze_map[i][j]
            if struct == 1:  # Unmovable block.
                # Offset all coordinates so that robot starts at the origin.
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f"
                    % (
                        j * maze_size_scaling - torso_x,
                        i * maze_size_scaling - torso_y,
                        maze_height / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f"
                    % (
                        0.5 * maze_size_scaling,
                        0.5 * maze_size_scaling,
                        maze_height / 2 * maze_size_scaling,
                    ),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    # Save new xml with maze to a temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_xml_path = path.join(path.dirname(tmp_dir), "ant_maze.xml")
        tree.write(temp_xml_path)

    return temp_xml_path


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
        maze_map=HARDEST_MAZE_EVAL_TEST,
        maze_size_scaling=4,
        non_zero_reset=False,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        ant_xml_file_path = path.join(
            path.dirname(sys.modules[AntEnv.__module__].__file__), "assets/ant.xml"
        )

        tmp_xml_file_path = add_maze_to_xml(
            ant_xml_file_path, maze_map, maze_size_scaling=maze_size_scaling
        )

        self.ant = AntEnv(
            xml_file=tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            **kwargs
        )

        self._maze_map = maze_map
        self._np_maze_map = np.array(deepcopy(maze_map))
        self._maze_size_scaling = maze_size_scaling
        self._non_zero_reset = non_zero_reset

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

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

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.set_target_goal()
        ant_obs, ant_info = self.ant.reset(seed=seed)
        obs_dict = self._get_obs(ant_obs)

        return obs_dict, ant_info

    def set_target_goal(self, goal_input=None):
        if goal_input is None:
            self.target_goal = self.sample_goal()
        else:
            self.target_goal = goal_input

        # print ('Target Goal: ', self.target_goal)
        # Make sure that the goal used in self._goal is also reset:
        self._goal = self.target_goal

    def sample_goal(self, only_free_cells=True):
        valid_cells = []
        goal_cells = []

        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                if self._maze_map[i][j] in [0, RESET, GOAL] or not only_free_cells:
                    valid_cells.append((i, j))
                if self._maze_map[i][j] == GOAL:
                    goal_cells.append((i, j))

        # If there is a 'goal' designated, use that. Otherwise, any valid cell can
        # be a goal.
        sample_choices = goal_cells if goal_cells else valid_cells
        cell = sample_choices[self.np_random.choice(len(sample_choices))]
        xy = self._rowcol_to_xy(cell, add_random_noise=True)

        random_x = (
            self.np_random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
        )
        random_y = (
            self.np_random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
        )

        xy = np.array([max(xy[0] + random_x, 0), max(xy[1] + random_y, 0)])

        return xy

    def _rowcol_to_xy(self, rowcol, add_random_noise=False):
        row, col = rowcol
        x = col * self._maze_size_scaling - self._init_torso_x
        y = row * self._maze_size_scaling - self._init_torso_y
        if add_random_noise:
            x = x + self.np_random.uniform(low=0, high=self._maze_size_scaling * 0.25)
            y = y + self.np_random.uniform(low=0, high=self._maze_size_scaling * 0.25)
        return (x, y)

    def _find_robot(self):
        structure = self._maze_map
        size_scaling = self._maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == RESET:
                    return j * size_scaling, i * size_scaling
        raise ValueError("No robot in maze specification.")

    def step(self, action):
        ant_obs, _, _, _, info = self.ant.step(action)
        obs = self._get_obs(ant_obs)

        terminated = self.compute_terminated(obs["achieved_goal"], self._goal, info)
        truncated = self.compute_truncated(obs["achieved_goal"], self._goal, info)

        reward = self.compute_reward(obs["achieved_goal"], self._goal, info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self, ant_obs):
        achieved_goal = ant_obs[:2]
        observation = ant_obs[2:]

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self._goal.copy(),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.reward_type == "dense":
            return -np.linalg.norm(desired_goal - achieved_goal)
        elif self.reward_type == "sparse":
            return 1.0 if np.linalg.norm(achieved_goal - desired_goal) <= 0.5 else 0.0

    def compute_terminated(self, achieved_goal, desired_goal, info):
        if not self.continuing_task:
            # If task is continuoing terminate the episode when the goal is reached
            return bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.5)
        else:
            return False

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return False

    def render(self):
        self.ant.render()
        # Set the viewer parameters at initialization of the renderer
        if self._render_initialized:
            self.ant.viewer.cam.distance = self.ant.model.stat.extent * 0.5
            self._render_initialized = False

    def _get_reset_location(
        self,
    ):
        prob = (1.0 - self._np_maze_map) / np.sum(1.0 - self._np_maze_map)
        prob_row = np.sum(prob, 1)
        row_sample = self.np_random.choice(
            np.arange(self._np_maze_map.shape[0]), p=prob_row
        )
        col_sample = self.np.random.choice(
            np.arange(self._np_maze_map.shape[1]),
            p=prob[row_sample] * 1.0 / prob_row[row_sample],
        )
        reset_location = self._rowcol_to_xy((row_sample, col_sample))

        # Add some random noise
        random_x = (
            self.np.random.uniform(low=0, high=0.5) * 0.5 * self._maze_size_scaling
        )
        random_y = (
            self.np.random.uniform(low=0, high=0.5) * 0.5 * self._maze_size_scaling
        )

        return (
            max(reset_location[0] + random_x, 0),
            max(reset_location[1] + random_y, 0),
        )
