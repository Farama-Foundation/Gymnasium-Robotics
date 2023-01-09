import numpy as np
from gymnasium import spaces

from gymnasium_robotics.envs.shadow_dexterous_hand import (
    MujocoManipulateEnv,
    MujocoPyManipulateEnv,
)


class MujocoManipulateTouchSensorsEnv(MujocoManipulateEnv):
    def __init__(
        self,
        target_position,
        target_rotation,
        target_position_range,
        reward_type,
        initial_qpos={},
        randomize_initial_position=True,
        randomize_initial_rotation=True,
        distance_threshold=0.01,
        rotation_threshold=0.1,
        n_substeps=20,
        relative_control=False,
        ignore_z_target_rotation=False,
        touch_visualisation="on_touch",
        touch_get_obs="sensordata",
        **kwargs,
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation

        """
        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]

        super().__init__(
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=target_position_range,
            reward_type=reward_type,
            initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position,
            randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold,
            n_substeps=n_substeps,
            relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
            **kwargs,
        )

        for (
            k,
            v,
        ) in (
            self._model_names.sensor_name2id.items()
        ):  # get touch sensor site names and their ids
            if "robot0:TS_" in k:
                self._touch_sensor_id_site_id.append(
                    (
                        v,
                        self._model_names.site_name2id[
                            k.replace("robot0:TS_", "robot0:T_")
                        ],
                    )
                )
                self._touch_sensor_id.append(v)

        if self.touch_visualisation == "off":  # set touch sensors rgba values
            for _, site_id in self._touch_sensor_id_site_id:
                self.model.site_rgba[site_id][3] = 0.0
        elif self.touch_visualisation == "always":
            pass

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

    def _render_callback(self):
        super()._render_callback()
        if self.touch_visualisation == "on_touch":
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                if self.data.sensordata[touch_sensor_id] != 0.0:
                    self.model.site_rgba[site_id] = self.touch_color
                else:
                    self.model.site_rgba[site_id] = self.notouch_color

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        object_qvel = self._utils.get_joint_qvel(self.model, self.data, "object:joint")

        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        touch_values = []  # get touch sensor readings. if there is one, set value to 1

        if self.touch_get_obs == "sensordata":
            touch_values = self.data.sensordata[self._touch_sensor_id]
        elif self.touch_get_obs == "boolean":
            touch_values = self.data.sensordata[self._touch_sensor_id] > 0.0
        elif self.touch_get_obs == "log":
            touch_values = np.log(self.data.sensordata[self._touch_sensor_id] + 1.0)
        observation = np.concatenate(
            [robot_qpos, robot_qvel, object_qvel, achieved_goal, touch_values]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }


class MujocoPyManipulateTouchSensorsEnv(MujocoPyManipulateEnv):
    def __init__(
        self,
        target_position,
        target_rotation,
        target_position_range,
        reward_type,
        initial_qpos={},
        randomize_initial_position=True,
        randomize_initial_rotation=True,
        distance_threshold=0.01,
        rotation_threshold=0.1,
        n_substeps=20,
        relative_control=False,
        ignore_z_target_rotation=False,
        touch_visualisation="on_touch",
        touch_get_obs="sensordata",
        **kwargs,
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation

        """
        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]

        super().__init__(
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=target_position_range,
            reward_type=reward_type,
            initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position,
            randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold,
            n_substeps=n_substeps,
            relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
            **kwargs,
        )

        for (
            k,
            v,
        ) in (
            self.sim.model._sensor_name2id.items()
        ):  # get touch sensor site names and their ids
            if "robot0:TS_" in k:
                self._touch_sensor_id_site_id.append(
                    (
                        v,
                        self.sim.model._site_name2id[
                            k.replace("robot0:TS_", "robot0:T_")
                        ],
                    )
                )
                self._touch_sensor_id.append(v)

        if self.touch_visualisation == "off":  # set touch sensors rgba values
            for _, site_id in self._touch_sensor_id_site_id:
                self.sim.model.site_rgba[site_id][3] = 0.0
        elif self.touch_visualisation == "always":
            pass

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

    def _render_callback(self):
        super()._render_callback()
        if self.touch_visualisation == "on_touch":
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                if self.sim.data.sensordata[touch_sensor_id] != 0.0:
                    self.sim.model.site_rgba[site_id] = self.touch_color
                else:
                    self.sim.model.site_rgba[site_id] = self.notouch_color

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel("object:joint")

        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        touch_values = []  # get touch sensor readings. if there is one, set value to 1

        if self.touch_get_obs == "sensordata":
            touch_values = self.sim.data.sensordata[self._touch_sensor_id]
        elif self.touch_get_obs == "boolean":
            touch_values = self.sim.data.sensordata[self._touch_sensor_id] > 0.0
        elif self.touch_get_obs == "log":
            touch_values = np.log(self.sim.data.sensordata[self._touch_sensor_id] + 1.0)

        observation = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                object_qvel,
                achieved_goal,
                touch_values,
            ]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }
