import os

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.hand.manipulate import (
    MujocoManipulateEnv,
    MujocoPyManipulateEnv,
)

# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join("hand", "manipulate_block_touch_sensors.xml")
MANIPULATE_EGG_XML = os.path.join("hand", "manipulate_egg_touch_sensors.xml")
MANIPULATE_PEN_XML = os.path.join("hand", "manipulate_pen_touch_sensors.xml")


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


class MujocoHandBlockTouchSensorsEnv(MujocoManipulateTouchSensorsEnv, EzPickle):
    """
    ### Description

    This environment was introduced in ["Using Tactile Sensing to Improve the Sample Efficiency and Performance of Deep Deterministic Policy Gradients for Simulated In-Hand Manipulation Tasks"](https://www.frontiersin.org/articles/10.3389/frobt.2021.538773/full).

    The environment is based on the same robot hand as in the `HandReach` environment, the [Shadow Dexterous Hand](https://www.shadowrobot.com/). The task to be solved is the same as in the `HandManipulateBlock` environment. However, in this case the environment observation also includes tactile sensory information.
    This is achieved by placing a total of 92 MuJoCo [touch sensors](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=touch#sensor-touch) in the palm and finger phalanxes of the hand. The sensors are created by specifying the tactile sensors’ active zones by so-called sites. Each site can be represented as
    either ellipsoid (for the finger tips) or box (for the phalanxes and palm sensors). When rendering the environment the sites are visualized as red and green transparent shapes attached to the hand model. If a body’s contact point falls within a site’s volume and involves a geometry attached to the same body as the site,
    the corresponding contact force is included in the sensor reading. Soft contacts do not influence the above computation except inasmuch as the contact point might move outside of the site, in which case if a contact point falls outside the sensor zone, but the normal ray intersects the sensor zone, it is also included.
    MuJoCo touch sensors only report normal forces using Minkowski Portal Refinement approach . The output of the contact sensor is a non-negative scalar value of type float that is computed as the sum of all contact normal forces that were included for this sensor in the current time step . Thus, each sensor of the 92 virtual
    touch sensors has a non-negative scalar value.

    The sensors are divided between the areas of the tip, middle, and lower phalanx of the forefinger, middle, ring, and little fingers. In addition to the areas of the three thumb phalanxes and the paml. The number of sensors are divided as follows in the different defined areas of the hand:

    | Functional areas of the hand model | Number of areas | Sensors-per-area | Total Sensors |
    | ---------------------------------- | --------------- | ---------------- | ------------- |
    | Lower phalanx of the fingers       | 4               | 7                | 28            |
    | Middle phalanx of the fingers      | 4               | 5                | 20            |
    | Tip phalanxes of the fingers       | 4               | 5                | 20            |
    | Thumb phalanxes                    | 3               | 5                | 15            |
    | Palm                               | 1               | 9                | 9             |

    When adding the sensors to the `HandManipulateBlock` environment there are two possible environment initializations depending on the type of data returned by the touch sensors. This data can be continuous values of external forces or a boolean value which is `True` if the sensor detects any contact force and `False` if not.
    This two types of environments can be initialized from the environment id variations of `HandManipulateBlock` by adding the `_ContinuousTouchSensors` string to the id if the touch sensors return continuous force values or `_BooleanTouchSensors` if the values are boolean.

    ##### Continuous Touch Sensor Environments:

    * `HandManipulateBlock_ContinuousTouchSensors-v1`
    * `HandManipulateBlockRotateZ_ContinuousTouchSensors-v1`
    * `HandManipulateBlockRotateParallel_ContinuousTouchSensors-v1`
    * `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1`
    * `HandManipulateBlockFull_ContinuousTouchSensors-v1`

    ##### Boolean Touch Sensor Environments:

    * `HandManipulateBlock_BooleanTouchSensors-v1`
    * `HandManipulateBlockRotateZ_BooleanTouchSensors-v1`
    * `HandManipulateBlockRotateParallel_BooleanTouchSensors-v1`
    * `HandManipulateBlockRotateXYZ_BooleanTouchSensors-v1`
    * `HandManipulateBlockFull_BooleanTouchSensors-v1`

    The `Action Space`, `Rewards`, `Starting State`, `Episode End`, and `Arguments` sections are the same as for the `HandManipulateBlock` environment and its variations.


    ### Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and block states, as well as information about the goal and touch sensors. The dictionary consists of the same 3 keys as the `HandManipulateBlock` environments (`observation`,`desired_goal`, and `achieved_goal`).
    However, the `ndarray` of the observation is now of shape `(153, )` instead of `(61, )` since the touch sensor information is added at the end of the array with shape `(92,)`.

    ### Version History

    * v1: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: the environment depends on `mujoco_py` which is no longer maintained.

    """

    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoManipulateTouchSensorsEnv.__init__(
            self,
            model_path=MANIPULATE_BLOCK_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type, **kwargs
        )


class MujocoHandEggTouchSensorsEnv(MujocoManipulateTouchSensorsEnv, EzPickle):
    """
    ### Description

    This environment was introduced in ["Using Tactile Sensing to Improve the Sample Efficiency and Performance of Deep Deterministic Policy Gradients for Simulated In-Hand Manipulation Tasks"](https://www.frontiersin.org/articles/10.3389/frobt.2021.538773/full).

    The environment is based on the same robot hand as in the `HandReach` environment, the [Shadow Dexterous Hand](https://www.shadowrobot.com/). The task to be solved is the same as in the `HandManipulateEgg` environment. However, in this case the environment observation also includes tactile sensory information.
    This is achieved by placing a total of 92 MuJoCo [touch sensors](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=touch#sensor-touch) in the palm and finger phalanxes of the hand. The sensors are created by specifying the tactile sensors’ active zones by so-called sites. Each site can be represented
    as either ellipsoid (for the finger tips) or box (for the phalanxes and palm sensors). When rendering the environment the sites are visualized as red and green transparent shapes attached to the hand model. If a body’s contact point falls within a site’s volume and involves a geometry attached to the same body as the
    site, the corresponding contact force is included in the sensor reading. Soft contacts do not influence the above computation except inasmuch as the contact point might move outside of the site, in which case if a contact point falls outside the sensor zone, but the normal ray intersects the sensor zone, it is also included.
    MuJoCo touch sensors only report normal forces using Minkowski Portal Refinement approach . The output of the contact sensor is a non-negative scalar value of type float that is computed as the sum of all contact normal forces that were included for this sensor in the current time step . Thus, each sensor of the 92 virtual
    touch sensors has a non-negative scalar value.

    The sensors are divided between the areas of the tip, middle, and lower phalanx of the forefinger, middle, ring, and little fingers. In addition to the areas of the three thumb phalanxes and the paml. The number of sensors are divided as follows in the different defined areas of the hand:

    | Functional areas of the hand model | Number of areas | Sensors-per-area | Total Sensors |
    | ---------------------------------- | --------------- | ---------------- | ------------- |
    | Lower phalanx of the fingers       | 4               | 7                | 28            |
    | Middle phalanx of the fingers      | 4               | 5                | 20            |
    | Tip phalanxes of the fingers       | 4               | 5                | 20            |
    | Thumb phalanxes                    | 3               | 5                | 15            |
    | Palm                               | 1               | 9                | 9             |

    When adding the sensors to the `HandManipulateEgg` environment there are two possible environment initializations depending on the type of data returned by the touch sensors. This data can be continuous values of external forces or a boolean value which is `True` if the sensor detects any contact force and `False` if not.
    This two types of environments can be initialized from the environment id variations of `HandManipulateEgg` by adding the `_ContinuousTouchSensors` string to the id if the touch sensors return continuous force values or `_BooleanTouchSensors` if the values are boolean.

    ##### Continuous Touch Sensor Environments:
    * `HandManipulateEgg_ContinuousTouchSensors-v1`
    * `HandManipulateEggRotate_ContinuousTouchSensors-v1`
    * `HandManipulateEggFull_ContinuousTouchSensors-v1`

    ##### Boolean Touch Sensor Environments:
    * `HandManipulateEgg_BooleanTouchSensors-v1`
    * `HandManipulateEggRotate_BooleanTouchSensors-v1`
    * `HandManipulateEggFull_BooleanTouchSensors-v1`

    The `Action Space`, `Rewards`, `Starting State`, `Episode End`, and `Arguments` are the same as for the `HandManipulateEgg` environment and its variations.


    ### Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and egg states, as well as information about the goal and touch sensors. The dictionary consists of the same 3 keys as the `HandManipulateEgg` environments (`observation`,`desired_goal`, and `achieved_goal`).
    However, the `ndarray` of the observation is now of shape `(153, )` instead of `(61, )` since the touch sensor information is added at the end of the array with shape `(92,)`.

    ### Version History

    * v1: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: the environment depends on `mujoco_py` which is no longer maintained.

    """

    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoManipulateTouchSensorsEnv.__init__(
            self,
            model_path=MANIPULATE_EGG_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type, **kwargs
        )


class MujocoHandPenTouchSensorsEnv(MujocoManipulateTouchSensorsEnv, EzPickle):
    """
    ### Description

    This environment was introduced in ["Using Tactile Sensing to Improve the Sample Efficiency and Performance of Deep Deterministic Policy Gradients for Simulated In-Hand Manipulation Tasks"](https://www.frontiersin.org/articles/10.3389/frobt.2021.538773/full).

    The environment is based on the same robot hand as in the `HandReach` environment, the [Shadow Dexterous Hand](https://www.shadowrobot.com/). The task to be solved is the same as in the `HandManipulatePen` environment. However, in this case the environment observation also includes tactile sensory information.
    This is achieved by placing a total of 92 MuJoCo [touch sensors](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=touch#sensor-touch) in the palm and finger phalanxes of the hand. The sensors are created by specifying the tactile sensors’ active zones by so-called sites. Each site can be represented
    as either ellipsoid (for the finger tips) or box (for the phalanxes and palm sensors). When rendering the environment the sites are visualized as red and green transparent shapes attached to the hand model. If a body’s contact point falls within a site’s volume and involves a geometry attached to the same body as the site,
    the corresponding contact force is included in the sensor reading. Soft contacts do not influence the above computation except inasmuch as the contact point might move outside of the site, in which case if a contact point falls outside the sensor zone, but the normal ray intersects the sensor zone, it is also included.
    MuJoCo touch sensors only report normal forces using Minkowski Portal Refinement approach . The output of the contact sensor is a non-negative scalar value of type float that is computed as the sum of all contact normal forces that were included for this sensor in the current time step . Thus, each sensor of the 92 virtual
    touch sensors has a non-negative scalar value.

    The sensors are divided between the areas of the tip, middle, and lower phalanx of the forefinger, middle, ring, and little fingers. In addition to the areas of the three thumb phalanxes and the paml. The number of sensors are divided as follows in the different defined areas of the hand:

    | Functional areas of the hand model | Number of areas | Sensors-per-area | Total Sensors |
    | ---------------------------------- | --------------- | ---------------- | ------------- |
    | Lower phalanx of the fingers       | 4               | 7                | 28            |
    | Middle phalanx of the fingers      | 4               | 5                | 20            |
    | Tip phalanxes of the fingers       | 4               | 5                | 20            |
    | Thumb phalanxes                    | 3               | 5                | 15            |
    | Palm                               | 1               | 9                | 9             |

    When adding the sensors to the `HandManipulatePen` environment there are two possible environment initializations depending on the type of data returned by the touch sensors. This data can be continuous values of external forces or a boolean value which is `True` if the sensor detects any contact force and `False` if not.
    This two types of environments can be initialized from the environment id variations of `HandManipulatePen` by adding the `_ContinuousTouchSensors` string to the id if the touch sensors return continuous force values or `_BooleanTouchSensors` if the values are boolean.

    ##### Continuous Touch Sensor Environments:
    * `HandManipulatePen_ContinuousTouchSensors-v1`
    * `HandManipulatePenRotate_ContinuousTouchSensors-v1`
    * `HandManipulatePenFull_ContinuousTouchSensors-v1`

    ##### Boolean Touch Sensor Environments:
    * `HandManipulatePen_BooleanTouchSensors-v1`
    * `HandManipulatePenRotate_BooleanTouchSensors-v1`
    * `HandManipulatePenFull_BooleanTouchSensors-v1`

    The `Action Space`, `Rewards`, `Starting State`, `Episode End`, and `Arguments` are the same as for the `HandManipulatePen` environment and its variations.


    ### Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and Pen states, as well as information about the goal and touch sensors. The dictionary consists of the same 3 keys as the `HandManipulatePen` environments (`observation`,`desired_goal`, and `achieved_goal`).
    However, the `ndarray` of the observation is now of shape `(153, )` instead of `(61, )` since the touch sensor information is added at the end of the array with shape `(92,)`.

    ### Version History

    * v1: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: the environment depends on `mujoco_py` which is no longer maintained.
    """

    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoManipulateTouchSensorsEnv.__init__(
            self,
            model_path=MANIPULATE_PEN_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False,
            reward_type=reward_type,
            ignore_z_target_rotation=True,
            distance_threshold=0.05,
            **kwargs,
        )
        EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type, **kwargs
        )


class MujocoPyHandBlockTouchSensorsEnv(MujocoPyManipulateTouchSensorsEnv, EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoPyManipulateTouchSensorsEnv.__init__(
            self,
            model_path=MANIPULATE_BLOCK_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type, **kwargs
        )


class MujocoPyHandEggTouchSensorsEnv(MujocoPyManipulateTouchSensorsEnv, EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoPyManipulateTouchSensorsEnv.__init__(
            self,
            model_path=MANIPULATE_EGG_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type, **kwargs
        )


class MujocoPyHandPenTouchSensorsEnv(MujocoPyManipulateTouchSensorsEnv, EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        touch_get_obs="sensordata",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoPyManipulateTouchSensorsEnv.__init__(
            self,
            model_path=MANIPULATE_PEN_XML,
            touch_get_obs=touch_get_obs,
            target_rotation=target_rotation,
            target_position=target_position,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False,
            reward_type=reward_type,
            ignore_z_target_rotation=True,
            distance_threshold=0.05,
            **kwargs,
        )

        EzPickle.__init__(
            self, target_position, target_rotation, touch_get_obs, reward_type, **kwargs
        )
