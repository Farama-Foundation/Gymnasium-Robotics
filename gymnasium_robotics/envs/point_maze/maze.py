import tempfile
import xml.etree.ElementTree as ET
from os import path

import numpy as np

from gymnasium_robotics.envs.ant_maze.maps import COMBINED, GOAL, RESET


class Maze:
    def __init__(self, maze_map: list, maze_size_scaling: float, maze_height: float):

        self._maze_map = maze_map
        self._maze_size_scaling = maze_size_scaling
        self._maze_height = maze_height

        self._unique_goal_locations = []
        self._unique_reset_locations = []
        self._combined_locations = []

        # Get the center cell position of the maze. This will be the origin
        self._map_length = len(maze_map)
        self._map_width = len(maze_map[0])
        self._x_map_center = np.ceil(self.map_width / 2) * maze_size_scaling
        self._y_map_center = np.ceil(self.map_length / 2) * maze_size_scaling

    @property
    def maze_map(self):
        return self._maze_map

    @property
    def maze_size_scaling(self):
        return self._maze_size_scaling

    @property
    def maze_height(self):
        return self._maze_height

    @property
    def unique_goal_locations(self):
        return self._unique_goal_locations

    @property
    def unique_reset_locations(self):
        return self._unique_reset_locations

    @property
    def combined_locations(self):
        return self._combined_locations

    @property
    def map_length(self):
        return self._map_length

    @property
    def map_width(self):
        return self._map_width

    @property
    def x_map_center(self):
        return self._x_map_center

    @property
    def y_map_center(self):
        return self._y_map_center

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        x = rowcol_pos[1] - self.x_map_center
        y = rowcol_pos[0] - self.y_map_center

        return np.array([x, y])

    @classmethod
    def make_maze(
        cls,
        agent_xml_path: str,
        maze_map: list,
        maze_size_scaling: float,
        maze_height: float,
    ):
        tree = ET.parse(agent_xml_path)
        worldbody = tree.find(".//worldbody")

        maze = cls(maze_map, maze_size_scaling, maze_height)

        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Store cell locations in simulation global Cartesian coordinates
                x = j * maze_size_scaling - maze.x_map_center
                y = i * maze_size_scaling - maze.y_map_center
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
                    print("RESET")
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif maze_map[i][j] in [
                    GOAL,
                ]:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif maze_map[i][j] in [
                    COMBINED,
                ]:
                    maze._combined_locations.append(np.array([x, y]))

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
        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Save new xml with maze to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_xml_path = path.join(path.dirname(tmp_dir), "ant_maze.xml")
            tree.write(temp_xml_path)

        return maze, temp_xml_path
