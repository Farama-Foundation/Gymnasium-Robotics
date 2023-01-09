import os

import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand import (
    MujocoManipulateTouchSensorsEnv,
    MujocoPyManipulateTouchSensorsEnv,
)

# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join("hand", "manipulate_block_touch_sensors.xml")


class MujocoHandBlockTouchSensorsEnv(MujocoManipulateTouchSensorsEnv, EzPickle):
    """
    ## Description

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

    #### Continuous Touch Sensor Environments:

    * `HandManipulateBlock_ContinuousTouchSensors-v1`
    * `HandManipulateBlockRotateZ_ContinuousTouchSensors-v1`
    * `HandManipulateBlockRotateParallel_ContinuousTouchSensors-v1`
    * `HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1`
    * `HandManipulateBlockFull_ContinuousTouchSensors-v1`

    #### Boolean Touch Sensor Environments:

    * `HandManipulateBlock_BooleanTouchSensors-v1`
    * `HandManipulateBlockRotateZ_BooleanTouchSensors-v1`
    * `HandManipulateBlockRotateParallel_BooleanTouchSensors-v1`
    * `HandManipulateBlockRotateXYZ_BooleanTouchSensors-v1`
    * `HandManipulateBlockFull_BooleanTouchSensors-v1`

    The `Action Space`, `Rewards`, `Starting State`, `Episode End`, and `Arguments` sections are the same as for the `HandManipulateBlock` environment and its variations.


    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and block states, as well as information about the goal and touch sensors. The dictionary consists of the same 3 keys as the `HandManipulateBlock` environments (`observation`,`desired_goal`, and `achieved_goal`).
    However, the `ndarray` of the observation is now of shape `(153, )` instead of `(61, )` since the touch sensor information is added at the end of the array with shape `(92,)`.

    ## Version History

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
