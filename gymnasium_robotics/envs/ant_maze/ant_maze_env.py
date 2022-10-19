import sys
import tempfile
import xml.etree.ElementTree as ET
from os import path
from typing import Dict, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v4 import AntEnv

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.ant_maze.maps import (
    COMBINED,
    GOAL,
    HARDEST_MAZE_TEST,
    RESET,
)
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames


def generate_maze(
    xml_path: str,
    maze_map: list,
    maze_size_scaling: float = 4,
    maze_height: float = 0.5,
):

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")
    unique_goal_locations = []
    unique_reset_locations = []
    combined_locations = []

    # Get the center cell position of the maze. This will be the origin
    map_length = len(maze_map)
    map_width = len(maze_map[0])
    x_map_center = np.ceil(map_width / 2) * maze_size_scaling
    y_map_center = np.ceil(map_length / 2) * maze_size_scaling

    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            struct = maze_map[i][j]
            # Store cell locations in simulation global Cartesian coordinates
            x = j * maze_size_scaling - x_map_center
            y = i * maze_size_scaling - y_map_center
            if struct == 1:  # Unmovable block.

                # Offset all coordinates so that maze is centered.
                ET.SubElement(
                    worldbody,
                    "geom",
                    name=f"block_{i}_{j}",
                    pos=f"{x} {y} {maze_height / 2 * maze_size_scaling}",
                    size=f"{0.5 * maze_size_scaling} {0.5 * maze_size_scaling} {maze_height / 2 * maze_size_scaling}",
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

            elif maze_map[i][j] in [
                RESET,
            ]:
                unique_reset_locations.append(np.array([x, y]))
            elif maze_map[i][j] in [
                GOAL,
            ]:
                unique_goal_locations.append(np.array([x, y]))
            elif maze_map[i][j] in [
                COMBINED,
            ]:
                combined_locations.append(np.array([x, y]))

    # Add target site for visualization
    ET.SubElement(
        worldbody,
        "site",
        name="target",
        pos=f"0 0 {maze_height / 2 * maze_size_scaling}",
        size=f"{0.1 * maze_size_scaling}",
        rgba="1 0 0 0.7",
        type="sphere",
    )

    # Save new xml with maze to a temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_xml_path = path.join(path.dirname(tmp_dir), "ant_maze.xml")
        tree.write(temp_xml_path)

    return (
        temp_xml_path,
        np.array([x_map_center, y_map_center]),
        unique_goal_locations,
        unique_reset_locations,
        combined_locations,
    )


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
        maze_map=HARDEST_MAZE_TEST,
        maze_size_scaling=4,
        maze_height=0.5,
        position_noise_range=0.25,
        non_zero_reset=False,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        # Get the ant.xml path from the Gymnasium package
        ant_xml_file_path = path.join(
            path.dirname(sys.modules[AntEnv.__module__].__file__), "assets/ant.xml"
        )
        # Add the maze to the xml
        (
            tmp_xml_file_path,
            self.map_center,
            self.goal_loc,
            self.reset_loc,
            self.comb_loc,
        ) = generate_maze(
            ant_xml_file_path,
            maze_map,
            maze_size_scaling=maze_size_scaling,
            maze_height=maze_height,
        )

        # Add the combined cell locations (goal/reset) to goal and reset
        self.goal_loc += self.comb_loc
        self.reset_loc += self.comb_loc

        # Create the MuJoCo environment, include position observation of the Ant for GoalEnv
        self.ant = AntEnv(
            xml_file=tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.ant.model)

        self._maze_height = maze_height
        self._maze_map = maze_map
        self._maze_size_scaling = maze_size_scaling
        self._non_zero_reset = non_zero_reset

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
        assert len(self.goal_loc) > 0
        goal_index = self.np_random.integers(low=0, high=len(self.goal_loc))
        goal = self.goal_loc[goal_index].copy()

        return goal

    def generate_reset_pos(self):
        assert len(self.reset_loc) > 0, ""

        # While reset position is close to goal position
        reset_pos = self.goal.copy()
        while np.linalg.norm(reset_pos - self.goal) <= 0.5:
            reset_index = self.np_random.integers(low=0, high=len(self.reset_loc))
            reset_pos = self.reset_loc[reset_index].copy()

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
            assert len(self._maze_map) > options["goal_cell"][1]
            assert len(self._maze_map[0]) > options["goal_cell"][0]
            assert (
                self._maze_map[options["goal_cell"][1], options["goal_cell"][0]] != 1
            ), f"Goal can't be placed in a wall cell, {options['goal_cell']}"

            goal = self.cell_rowcol_to_xy(options["goal_cell"])

        else:
            goal = self.generate_target_goal()

        # Add noise to goal position
        self.goal = self.add_xy_position_noise(goal)

        # Update target site for visualization
        site_id = self._model_names.site_name2id["target"]
        self.ant.model.site_pos[site_id] = np.append(
            self.goal, self._maze_height / 2 * self._maze_size_scaling
        )

        if options["reset_cell"] is not None:
            # assert that goal cell is valid
            assert len(self._maze_map) > options["reset_cell"][1]
            assert len(self._maze_map[0]) > options["reset_cell"][0]
            assert (
                self._maze_map[options["reset_cell"][1], options["reset_cell"][0]] != 1
            ), f"Reset can't be placed in a wall cell, {options['reset_cell']}"

            reset_pos = self.cell_rowcol_to_xy(options["reset_cell"])

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

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        x = rowcol_pos[1] - self.map_center[0]
        y = rowcol_pos[0] - self.map_center[1]

        return np.array([x, y])

    def add_xy_position_noise(self, xy_pos: np.ndarray) -> np.ndarray:
        noise_x = self.np_random.uniform(low=-0.25, high=0.25) * self._maze_size_scaling
        noise_y = self.np_random.uniform(low=-0.25, high=0.25) * self._maze_size_scaling
        xy_pos[0] += noise_x
        xy_pos[1] += noise_y

        return xy_pos
