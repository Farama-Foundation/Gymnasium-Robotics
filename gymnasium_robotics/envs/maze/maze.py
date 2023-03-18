"""A maze environment with Gymnasium API for the Gymnasium-Robotics PointMaze environments.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files: `maps.py`, `maze_env.py`, `point_env.py`, and `point_maze_env.py`.
As well as adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""
import math
import tempfile
import xml.etree.ElementTree as ET
from os import path
from typing import Dict, List, Optional, Union

import numpy as np

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.maze.maps import COMBINED, GOAL, RESET, U_MAZE


class Maze:
    r"""This class creates and holds information about the maze in the MuJoCo simulation.

    The accessible attributes are the following:
    - :attr:`maze_map` - The maze discrete data structure.
    - :attr:`maze_size_scaling` - The maze scaling for the continuous coordinates in the MuJoCo simulation.
    - :attr:`maze_height` - The height of the walls in the MuJoCo simulation.
    - :attr:`unique_goal_locations` - All the `(i,j)` possible cell indices for goal locations.
    - :attr:`unique_reset_locations` - All the `(i,j)` possible cell indices for agent initialization locations.
    - :attr:`combined_locations` - All the `(i,j)` possible cell indices for goal and agent initialization locations.
    - :attr:`map_length` - Maximum value of j cell index
    - :attr:`map_width` - Mazimum value of i cell index
    - :attr:`x_map_center` - The x coordinate of the map's center
    - :attr:`y_map_center` - The y coordinate of the map's center

    The Maze class also presents a method to convert from cell indices to `(x,y)` coordinates in the MuJoCo simulation:
    - :meth:`cell_rowcol_to_xy` - Convert from `(i,j)` to `(x,y)`

    """

    def __init__(
        self,
        maze_map: List[List[Union[str, int]]],
        maze_size_scaling: float,
        maze_height: float,
    ):

        self._maze_map = maze_map
        self._maze_size_scaling = maze_size_scaling
        self._maze_height = maze_height

        self._unique_goal_locations = []
        self._unique_reset_locations = []
        self._combined_locations = []

        # Get the center cell Cartesian position of the maze. This will be the origin
        self._map_length = len(maze_map)
        self._map_width = len(maze_map[0])
        self._x_map_center = self.map_width / 2 * maze_size_scaling
        self._y_map_center = self.map_length / 2 * maze_size_scaling

    @property
    def maze_map(self) -> List[List[Union[str, int]]]:
        """Returns the list[list] data structure of the maze."""
        return self._maze_map

    @property
    def maze_size_scaling(self) -> float:
        """Returns the scaling value used to integrate the maze
        encoding in the MuJoCo simulation.
        """
        return self._maze_size_scaling

    @property
    def maze_height(self) -> float:
        """Returns the un-scaled height of the walls in the MuJoCo
        simulation.
        """
        return self._maze_height

    @property
    def unique_goal_locations(self) -> List[np.ndarray]:
        """Returns all the possible goal locations in discrete cell
        coordinates (i,j)
        """
        return self._unique_goal_locations

    @property
    def unique_reset_locations(self) -> List[np.ndarray]:
        """Returns all the possible reset locations for the agent in
        discrete cell coordinates (i,j)
        """
        return self._unique_reset_locations

    @property
    def combined_locations(self) -> List[np.ndarray]:
        """Returns all the possible goal/reset locations in discrete cell
        coordinates (i,j)
        """
        return self._combined_locations

    @property
    def map_length(self) -> int:
        """Returns the length of the maze in number of discrete vertical cells
        or number of rows i.
        """
        return self._map_length

    @property
    def map_width(self) -> int:
        """Returns the width of the maze in number of discrete horizontal cells
        or number of columns j.
        """
        return self._map_width

    @property
    def x_map_center(self) -> float:
        """Returns the x coordinate of the center of the maze in the MuJoCo simulation"""
        return self._x_map_center

    @property
    def y_map_center(self) -> float:
        """Returns the x coordinate of the center of the maze in the MuJoCo simulation"""
        return self._y_map_center

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        """Converts a cell index `(i,j)` to x and y coordinates in the MuJoCo simulation"""
        x = (rowcol_pos[1] + 0.5) * self.maze_size_scaling - self.x_map_center
        y = self.y_map_center - (rowcol_pos[0] + 0.5) * self.maze_size_scaling

        return np.array([x, y])

    def cell_xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        """Converts a cell x and y coordinates to `(i,j)`"""
        i = math.floor((self.y_map_center - xy_pos[1]) / self.maze_size_scaling)
        j = math.floor((xy_pos[0] + self.x_map_center) / self.maze_size_scaling)
        return np.array([i, j])

    @classmethod
    def make_maze(
        cls,
        agent_xml_path: str,
        maze_map: list,
        maze_size_scaling: float,
        maze_height: float,
    ):
        """Class method that returns an instance of Maze with a decoded maze information and the temporal
           path to the new MJCF (xml) file for the MuJoCo simulation.

        Args:
            agent_xml_path (str): the goal that was achieved during execution
            maze_map (list[list[str,int]]): the desired goal that we asked the agent to attempt to achieve
            maze_size_scaling (float): an info dictionary with additional information
            maze_height (float): an info dictionary with additional information

        Returns:
            Maze: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
            str: The xml temporal file to the new mjcf model with the included maze.
        """
        tree = ET.parse(agent_xml_path)
        worldbody = tree.find(".//worldbody")

        maze = cls(maze_map, maze_size_scaling, maze_height)
        empty_locations = []
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * maze_size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * maze_size_scaling
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

                elif struct == RESET:
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif struct == GOAL:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    maze._combined_locations.append(np.array([x, y]))
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))

        # Add target site for visualization
        ET.SubElement(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 {maze_height / 2 * maze_size_scaling}",
            size=f"{0.2 * maze_size_scaling}",
            rgba="1 0 0 0.7",
            type="sphere",
        )

        # Add the combined cell locations (goal/reset) to goal and reset
        if (
            not maze._unique_goal_locations
            and not maze._unique_reset_locations
            and not maze._combined_locations
        ):
            # If there are no given "r", "g" or "c" cells in the maze data structure,
            # any empty cell can be a reset or goal location at initialization.
            maze._combined_locations = empty_locations
        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_path = path.join(path.dirname(tmp_dir), "ant_maze.xml")
            tree.write(temp_xml_path)

        return maze, temp_xml_path


class MazeEnv(GoalEnv):
    def __init__(
        self,
        agent_xml_path: str,
        reward_type: str = "dense",
        continuing_task: bool = True,
        maze_map: List[List[Union[int, str]]] = U_MAZE,
        maze_size_scaling: float = 1.0,
        maze_height: float = 0.5,
        position_noise_range: float = 0.25,
        **kwargs,
    ):

        self.reward_type = reward_type
        self.continuing_task = continuing_task
        self.maze, self.tmp_xml_file_path = Maze.make_maze(
            agent_xml_path, maze_map, maze_size_scaling, maze_height
        )

        self.position_noise_range = position_noise_range

    def generate_target_goal(self) -> np.ndarray:
        assert len(self.maze.unique_goal_locations) > 0
        goal_index = self.np_random.integers(
            low=0, high=len(self.maze.unique_goal_locations)
        )
        goal = self.maze.unique_goal_locations[goal_index].copy()
        return goal

    def generate_reset_pos(self) -> np.ndarray:
        assert len(self.maze.unique_reset_locations) > 0

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
        options: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):
        super().reset(seed=seed)

        if options is None:
            goal = self.generate_target_goal()
            # Add noise to goal position
            self.goal = self.add_xy_position_noise(goal)
            reset_pos = self.generate_reset_pos()
        else:
            if "goal_cell" in options and options["goal_cell"] is not None:
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

            if "reset_cell" in options and options["reset_cell"] is not None:
                # assert that goal cell is valid
                assert self.maze.map_length > options["reset_cell"][1]
                assert self.maze.map_width > options["reset_cell"][0]
                assert (
                    self.maze.maze_map[
                        options["reset_cell"][1], options["reset_cell"][0]
                    ]
                    != 1
                ), f"Reset can't be placed in a wall cell, {options['reset_cell']}"

                reset_pos = self.maze.cell_rowcol_to_xy(options["reset_cell"])

            else:
                reset_pos = self.generate_reset_pos()

        # Update the position of the target site for visualization
        self.update_target_site_pos()
        # Add noise to reset position
        self.reset_pos = self.add_xy_position_noise(reset_pos)

        # Update the position of the target site for visualization
        self.update_target_site_pos()

    def add_xy_position_noise(self, xy_pos: np.ndarray) -> np.ndarray:
        """Pass an x,y coordinate and it will return the same coordinate with a noise addition
        sampled from a uniform distribution
        """
        noise_x = (
            self.np_random.uniform(
                low=-self.position_noise_range, high=self.position_noise_range
            )
            * self.maze.maze_size_scaling
        )
        noise_y = (
            self.np_random.uniform(
                low=-self.position_noise_range, high=self.position_noise_range
            )
            * self.maze.maze_size_scaling
        )
        xy_pos[0] += noise_x
        xy_pos[1] += noise_y

        return xy_pos

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> float:
        if self.reward_type == "dense":
            return np.exp(-np.linalg.norm(desired_goal - achieved_goal))
        elif self.reward_type == "sparse":
            return 1.0 if np.linalg.norm(achieved_goal - desired_goal) <= 0.45 else 0.0

    def compute_terminated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        if not self.continuing_task:
            # If task is episodic terminate the episode when the goal is reached
            return bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.45)
        else:
            # Continuing tasks don't terminate, episode will be truncated when time limit is reached (`max_episode_steps`)
            if (
                bool(np.linalg.norm(achieved_goal - desired_goal) <= 0.45)
                and len(self.maze.unique_goal_locations) > 1
            ):
                # Generate another goal
                goal = self.generate_target_goal()
                # Add noise to goal position
                self.goal = self.add_xy_position_noise(goal)
                # Update the position of the target site for visualization
                self.update_target_site_pos()

            return False

    def compute_truncated(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> bool:
        return False

    def update_target_site_pos(self, pos):
        """Override this method to update the site qpos in the MuJoCo simulation
        after a new goal is selected. This is mainly for visualization purposes."""
        raise NotImplementedError
