"""Environment using Gymnasium API and Multi-goal API for kitchen and Franka robot.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

This code was also implemented over the repository relay-policy-learning on GitHub (https://github.com/google-research/relay-policy-learning),
published in Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning, by
Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman.

Original Author of the code: Abhishek Gupta & Justin Fu

The modifications made involve separatin the Kitchen environment from the Franka environment and addint support for compatibility with
the Gymnasium and Multi-goal API's.

This project is covered by the Apache 2.0 License.
"""

from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.franka_kitchen.franka_env import FrankaRobot
from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos

OBS_ELEMENT_GOALS = {
    "bottom_right_burner": np.array([-0.01]),
    "bottom_left_burner": np.array([-0.01]),
    "top_right_burner": np.array([-0.01]),
    "top_left_burner": np.array([-0.01]),
    "light_switch": np.array([-0.7]),
    "slide_cabinet": np.array([0.37]),
    "left_hinge_cabinet": np.array([-1.45]),
    "right_hinge_cabinet": np.array([1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3


class KitchenEnv(GoalEnv, EzPickle):
    """
    ## Description

    This environment was introduced in ["Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning"](https://arxiv.org/abs/1910.11956)
    by Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, Karol Hausman.

    The environment is based on the 9 degrees of freedom [Franka robot](https://www.franka.de/). The Franka robot is placed in a kitchen environment containing several common
    household items: a microwave, a kettle, an overhead light, cabinets, and an oven. The environment is a `multitask` goal in which the robot has to interact with the previously
    mentioned items in order to reach a desired goal configuration. For example, one such state is to have the microwave andv sliding cabinet door open with the kettle on the top burner
    and the overhead light on. The final tasks to be included in the goal can be configured when the environment is initialized.

    Also, some updates have been done to the original MuJoCo model, such as using an inverse kinematic controller and substituting the legacy Franka model with the maintained MuJoCo model
    in [deepmind/mujoco_menagerie](https://github.com/deepmind/mujoco_menagerie).

    ## Goal

    The goal has a multitask configuration. The multiple tasks to be completed in an episode can be set by passing a list of tasks to the argument`tasks_to_complete`. For example, to open
    the microwave door and move the kettle create the environment as follows:

    ```python
    import gymnasium as gym
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'])
    ```

    The following is a table with all the possible tasks and their respective joint goal values:

    | Task (Joint Name in XML) | Description                                                    | Joint Type | Goal                                       |
    | ------------------------ | -------------------------------------------------------------- | ---------- | ------------------------------------------ |
    | "bottom_right_burner"    | Turn the knob oven knob that activates the bottom right burner | slide      | -0.01                                      |
    | "bottom_left_burner"     | Turn the knob oven knob that activates the bottom right burner | slide      | -0.01                                      |
    | "top_right_burner"       | Turn the knob oven knob that activates the bottom right burner | slide      | -0.01                                      |
    | "top_left_burner"        | Turn the knob oven knob that activates the bottom right burner | slide      | -0.01                                      |
    | light_switch"            | Turn on the light switch                                       | hinge      | -0.7                                       |
    | "slide_cabinet"          | Open the slide cabinet                                         | hinge      | 0.37                                       |
    | "left_hinge_cabinet"     | Open the left hinge cabinet                                    | hinge      | -1.45                                      |
    | "right_hinge_cabinet"    | Open the right hinge cabinet                                   | hinge      | 1.45                                       |
    | "microwave"              | Open the microwave door                                        | hinge      | -0.75                                      |
    | "kettle"                 | Move the kettle to the top left burner                         | free       | [-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06] |


    ## Action Space

    The action space of this environment is different from its original implementation in ["Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning"](https://arxiv.org/abs/1910.11956).
    The environment can be initialized with two possible robot controls: an end-effector `inverse kinematic controller` or a `joint position controller`.

    ### IK Controller
    This action space can be used by setting the argument `ik_controller` to `True`. The space is a `Box(-1.0, 1.0, (7,), float32)`, which represents the end-effector's desired pose displacement.
    The end-effector's frame of reference is defined with a MuJoCo site named `EEF`. The controller relies on the model's position control of the actuators by using the Damped Least Squares (DLS)
    algorithm to compute displacements in the joint space from a desired end-effector configuration. The controller iteratively reduces the error from the current end-effector pose with respect
    to the desired target. The default number of controller steps per Gymnasium step is `5`. This value can be set with the `control_steps` argument, however it is not recommended to reduce this value
    since the simulation can get unstable.

    | Num | Action                                                                            | Action Min  | Action Max  | Control Min  | Control Max | Unit        |
    | --- | ----------------------------------------------------------------------------------| ----------- | ----------- | ------------ | ----------  | ----------- |
    | 0   | Linear displacement of the end-effector's current position in the x direction     | -1          | 1           | -0.2 (m)     | 0.2 (m)     | position (m)|
    | 1   | Linear displacement of the end-effector's current position in the y direction     | -1          | 1           | -0.2 (m)     | 0.2 (m)     | position (m)|
    | 2   | Linear displacement of the end-effector's current position in the z direction     | -1          | 1           | -0.2 (m)     | 0.2 (m)     | position (m)|
    | 3   | Angular displacement of the end-effector's current orientation around the x axis  | -1          | 1           | -0.5 (rad)   | 0.5 (rad)   | angle (rad) |
    | 4   | Angular displacement of the end-effector's current orientation around the y axis  | -1          | 1           | -0.5 (rad)   | 0.5 (rad)   | angle (rad) |
    | 5   | Angular displacement of the end-effector's current orientation around the z axis  | -1          | 1           | -0.5 (rad)   | 0.5 (rad)   | angle (rad) |
    | 6   | Force applied to the slide joint of the gripper                                   | -1          | 1           | 0.0 (N)      | 255 (N)     | force (N)   |

    ### Joint Position Controller
    The default control actuators in the Franka MuJoCo model are joint position controllers. The input action is denormalize from `[-1,1]` to the actuator range and input
    directly to the model. The space is a `Box(-1.0, 1.0, (8,), float32)`, and it can be used by setting the argument `ik_controller` to `False`.

    | Num | Action                                              | Action Min  | Action Max  | Control Min  | Control Max | Name (in corresponding XML file) | Joint | Unit        |
    | --- | ----------------------------------------------------| ----------- | ----------- | -----------  |-------------| ---------------------------------| ----- | ----------- |
    | 0   | `robot:joint1` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator1                        | hinge | angle (rad) |
    | 1   | `robot:joint2` angular position                     | -1          | 1           | -1.76 (rad)  | 1.76 (rad)  | actuator2                        | hinge | angle (rad) |
    | 2   | `robot:joint3` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator3                        | hinge | angle (rad) |
    | 3   | `robot:joint4` angular position                     | -1          | 1           | -3.07 (rad)  | 0.0 (rad)   | actuator4                        | hinge | angle (rad) |
    | 4   | `robot:joint5` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator5                        | hinge | angle (rad) |
    | 5   | `robot:joint6` angular position                     | -1          | 1           | -0.0 (rad)   | 3.75 (rad)  | actuator6                        | hinge | angle (rad) |
    | 6   | `robot:joint7` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator7                        | hinge | angle (rad) |
    | 7   | `robot:finger_left` and `robot:finger_right` force  | -1          | 1           | 0 (N)        | 255 (N)     | actuator8                        | slide | force (N)   |

    ## Observation Space

    The observation is a `goal-aware observation space`. The observation space contains the following keys:

    * `observation`: this is a `Box(-inf, inf, shape=(59,), dtype="float64")` space and it is formed by the robot's joint positions and velocities, as well as
        the pose and velocities of the kitchen items. An additional uniform noise of ranfe `[-1,1]` is added to the observations. The noise is also scaled by a factor
        of `robot_noise_ratio` and `object_noise_ratio` given in the environment arguments. The elements of the `observation` array are the following:


    | Num   | Observation                                           | Min      | Max      | Joint Name (in corresponding XML file)   | Joint Type | Unit                       |
    | ----- | ----------------------------------------------------- | -------- | -------- | ---------------------------------------- | ---------- | -------------------------- |
    | 0     | `robot:joint1` hinge joint angle value                | -Inf     | Inf      | robot:joint1                             | hinge      | angle (rad)                |
    | 1     | `robot:joint2` hinge joint angle value                | -Inf     | Inf      | robot:joint2                             | hinge      | angle (rad)                |
    | 2     | `robot:joint3` hinge joint angle value                | -Inf     | Inf      | robot:joint3                             | hinge      | angle (rad)                |
    | 3     | `robot:joint4` hinge joint angle value                | -Inf     | Inf      | robot:joint4                             | hinge      | angle (rad)                |
    | 4     | `robot:joint5` hinge joint angle value                | -Inf     | Inf      | robot:joint5                             | hinge      | angle (rad)                |
    | 5     | `robot:joint6` hinge joint angle value                | -Inf     | Inf      | robot:joint6                             | hinge      | angle (rad)                |
    | 6     | `robot:joint7` hinge joint angle value                | -Inf     | Inf      | robot:joint7                             | hinge      | angle (rad)                |
    | 7     | `robot:finger_joint1` slide joint translation value   | -Inf     | Inf      | robot:finger_joint1                      | slide      | position (m)               |
    | 8     | `robot:finger_joint2` slide joint translation value   | -Inf     | Inf      | robot:finger_joint2                      | slide      | position (m)               |
    | 9     | `robot:joint1` hinge joint angular velocity           | -Inf     | Inf      | robot:joint1                             | hinge      | angular velocity (rad/s)   |
    | 10    | `robot:joint2` hinge joint angular velocity           | -Inf     | Inf      | robot:joint2                             | hinge      | angular velocity (rad/s)   |
    | 11    | `robot:joint3` hinge joint angular velocity           | -Inf     | Inf      | robot:joint3                             | hinge      | angular velocity (rad/s)   |
    | 12    | `robot:joint4` hinge joint angular velocity           | -Inf     | Inf      | robot:joint4                             | hinge      | angular velocity (rad/s)   |
    | 13    | `robot:joint5` hinge joint angular velocity           | -Inf     | Inf      | robot:joint5                             | hinge      | angular velocity (rad/s)   |
    | 14    | `robot:joint6` hinge joint angular velocity           | -Inf     | Inf      | robot:joint6                             | hinge      | angular velocity (rad/s)   |
    | 15    | `robot:joint7` hinge joint angular velocity           | -Inf     | Inf      | robot:joint7                             | hinge      | angle (rad)                |
    | 16    | `robot:finger_joint1` slide joint linear velocity     | -Inf     | Inf      | robot:finger_joint1                      | slide      | linear velocity (m/s)      |
    | 17    | `robot:finger_joint2` slide joint linear velocity     | -Inf     | Inf      | robot:finger_joint2                      | slide      | linear velocity (m/s)      |
    | 18    | Rotation of the knob for the bottom right burner      | -Inf     | Inf      | knob_Joint_1                             | hinge      | angle (rad)                |
    | 19    | Joint opening of the bottom right burner              | -Inf     | Inf      | bottom_right_burner                      | slide      | position (m)               |
    | 20    | Rotation of the knob for the bottom left burner       | -Inf     | Inf      | knob_Joint_2                             | hinge      | angle (rad)                |
    | 21    | Joint opening of the bottom left burner               | -Inf     | Inf      | bottom_left_burner                       | slide      | position (m)               |
    | 22    | Rotation of the knob for the top right burner         | -Inf     | Inf      | knob_Joint_3                             | hinge      | angle (rad)                |
    | 23    | Joint opening of the top right burner                 | -Inf     | Inf      | top_right_burner                         | slide      | position (m)               |
    | 24    | Rotation of the knob for the top left burner          | -Inf     | Inf      | knob_Joint_4                             | hinge      | angle (rad)                |
    | 25    | Joint opening of the top left burner                  | -Inf     | Inf      | top_left_burner                          | slide      | position (m)               |
    | 26    | Joint angle value of the overhead light switch        | -Inf     | Inf      | light_switch                             | slide      | position (m)               |
    | 27    | Opening of the overhead light joint                   | -Inf     | Inf      | light_joint                              | hinge      | angle (rad)                |
    | 28    | Translation of the slide cabinet joint                | -Inf     | Inf      | slide_cabinet                            | slide      | position (m)               |
    | 29    | Rotation of the joint in the left hinge cabinet       | -Inf     | Inf      | left_hinge_cabinet                       | hinge      | angle (rad)                |
    | 30    | Rotation of the joint in the right hinge cabinet      | -Inf     | Inf      | right_hinge_cabinet                      | hinge      | angle (rad)                |
    | 31    | Rotation of the joint in the microwave door           | -Inf     | Inf      | microwave                                | hinge      | angle (rad)                |
    | 32    | Kettle's x coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 33    | Kettle's y coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 34    | Kettle's z coordinate                                 | -Inf     | Inf      | kettle                                   | free       | position (m)               |
    | 35    | Kettle's x quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 36    | Kettle's y quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 37    | Kettle's z quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 38    | Kettle's w quaternion rotation                        | -Inf     | Inf      | kettle                                   | free       | -                          |
    | 39    | Bottom right burner knob angular velocity             | -Inf     | Inf      | knob_Joint_1                             | hinge      | angular velocity (rad/s)   |
    | 40    | Opening linear velocity  of the bottom right burner   | -Inf     | Inf      | bottom_right_burner                      | slide      | velocity (m/s)             |
    | 41    | Bottom left burner knob angular velocity              | -Inf     | Inf      | knob_Joint_2                             | hinge      | angular velocity (rad/s)   |
    | 42    | Opening linear velocity of the bottom left burner     | -Inf     | Inf      | bottom_left_burner                       | slide      | velocity (m/s)             |
    | 43    | Top right burner knob angular velocity                | -Inf     | Inf      | knob_Joint_3                             | hinge      | angular velocity (rad/s)   |
    | 44    | Opening linear velocity of the top right burner       | -Inf     | Inf      | top_right_burner                         | slide      | velocity (m/s)             |
    | 45    | Top left burner knob angular velocity                 | -Inf     | Inf      | knob_Joint_4                             | hinge      | angular velocity (rad/s)   |
    | 46    | Opening linear velocity of the top left burner        | -Inf     | Inf      | top_left_burner                          | slide      | velocity (m/s)             |
    | 47    | Angular velocity of the overhead light switch         | -Inf     | Inf      | light_switch                             | slide      | velocity (m/s)             |
    | 48    | Opening linear velocity of the overhead light         | -Inf     | Inf      | light_joint                              | hinge      | angular velocity (rad/s)   |
    | 49    | Linear velocity of the slide cabinet joint            | -Inf     | Inf      | slide_cabinet                            | slide      | velocity (m/s)             |
    | 50    | Angular velocity of the left hinge cabinet joint      | -Inf     | Inf      | left_hinge_cabinet                       | hinge      | angular velocity (rad/s)   |
    | 51    | Angular velocity of the right hinge cabinet joint     | -Inf     | Inf      | right_hinge_cabinet                      | hinge      | angular velocity (rad/s)   |
    | 52    | Anular velocity of the microwave door joint           | -Inf     | Inf      | microwave                                | hinge      | angular velocity (rad/s)   |
    | 53    | Kettle's x linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 54    | Kettle's y linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 55    | Kettle's z linear velocity                            | -Inf     | Inf      | kettle                                   | free       | linear velocity (m/s)      |
    | 56    | Kettle's x axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |
    | 57    | Kettle's y axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |
    | 58    | Kettle's z axis angular rotation                      | -Inf     | Inf      | kettle                                   | free       | angular velocity(rad/s)    |

    * `desired_goal`: this key represents the final goal to be achieved. The value is another `Dict` space with keys the tasks to be completed in the episode and values the joint
    goal configuration of each joint in the task as specified in the `Goal` section.

    * `achieved_goal`: this key represents the current state of the tasks. The value is another `Dict` space with keys the tasks to be completed in the episode and values the
    current joint configuration of each joint in the task.

    ## Info

    The environment also returns an `info` dictionary in each Gymnasium step. The keys are:

    - `tasks_to_complete` (list[str]): list of tasks that haven't yet been completed in the current episode.
    - `step_task_completions` (list[str]): list of tasks completed in the step taken.
    - `episode_task_completions` (list[str]): list of tasks completed during the episode uptil the current step.

    ## Rewards

    The environment's reward is `sparser`. The reward in each Gymnasium step is equal to the number of task completed in the given step. If no task is completed the returned reward will be zero.
    The tasks are considered completed when their joint configuration is within a norm threshold of `0.3` with respect to the goal configuration specified in the `Goal` section.

    ## Starting State

    The simulation starts with all of the joint position actuators of the Franka robot set to zero. The doors of the microwave and cabinets are closed, the burners turned off, and the light switch also off. The kettle
    will be placed in the bottom left burner.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 280 timesteps.
    The episode is `terminated` when all the tasks have been completed unless the `terminate_on_tasks_completed` argument is set to `False`.

    ## Arguments

    The following arguments can be passed when initializing the environment with `gymnasium.make` kwargs:

    | Parameter                      | Type            | Default                                     | Description                                                                                                                                                               |
    | -------------------------------| --------------- | ------------------------------------------- | ----------------------------------------------------------------------------------- |
    | `tasks_to_complete`            | **list[str]**   | All possible goal tasks. Go to Goal section | The goal tasks to reach in each episode                                             |
    | `terminate_on_tasks_completed` | **bool**        | `True`                                      | Terminate episode if no more tasks to complete (episodic multitask)                 |
    | `remove_task_when_completed`   | **bool**        | `True`                                      | Remove the completed tasks from the info dictionary returned after each step        |
    | `object_noise_ratio`           | **float**       | `0.0005`                                    | Scaling factor applied to the uniform noise added to the kitchen object observations|
    | `ik_controller`                | **bool**        | `True`                                      | Use inverse kinematic (IK) controller or joint position controller                  |
    | `control_steps`                | **integer**     | `5`                                         | Number of steps to iterate the DLS in the IK controller                             |
    | `robot_noise_ratio`            | **float**       | `0.01`                                      | Scaling factor applied to the uniform noise added to the robot joint observations   |
    | `max_episode_steps`            | **integer**     | `280`                                       | Maximum number of steps per episode                                                 |

    ## Version History

    * v1: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
        self,
        tasks_to_complete: "list[str]" = list(OBS_ELEMENT_GOALS.keys()),
        terminate_on_tasks_completed: bool = True,
        remove_task_when_completed: bool = True,
        object_noise_ratio: float = 0.0005,
        **kwargs,
    ):
        self.robot_env = FrankaRobot(
            model_path="../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml",
            **kwargs,
        )

        self.robot_env.init_qpos[:7] = np.array(
            [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.0]
        )
        self.model = self.robot_env.model
        self.data = self.robot_env.data
        self.render_mode = self.robot_env.render_mode

        self.terminate_on_tasks_completed = terminate_on_tasks_completed
        self.remove_task_when_completed = remove_task_when_completed

        self.goal = {}
        self.tasks_to_complete = set(tasks_to_complete)
        # Validate list of tasks to complete
        for task in tasks_to_complete:
            if task not in OBS_ELEMENT_GOALS.keys():
                raise ValueError(
                    f"The task {task} cannot be found the the list of possible goals: {OBS_ELEMENT_GOALS.keys()}"
                )
            else:
                self.goal[task] = OBS_ELEMENT_GOALS[task]

        self.step_task_completions = (
            []
        )  # Tasks completed in the current environment step
        self.episode_task_completions = (
            []
        )  # Tasks completed that have been completed in the current episode
        self.object_noise_ratio = (
            object_noise_ratio  # stochastic noise added to the object observations
        )

        robot_obs = self.robot_env._get_obs()
        obs = self._get_obs(robot_obs)

        assert (
            int(np.round(1.0 / self.robot_env.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.action_space = self.robot_env.action_space
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Dict(
                    {
                        task: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=goal.shape,
                            dtype="float64",
                        )
                        for task, goal in obs["achieved_goal"].items()
                    }
                ),
                achieved_goal=spaces.Dict(
                    {
                        task: spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=goal.shape,
                            dtype="float64",
                        )
                        for task, goal in obs["achieved_goal"].items()
                    }
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

        EzPickle.__init__(
            self,
            tasks_to_complete,
            terminate_on_tasks_completed,
            remove_task_when_completed,
            object_noise_ratio,
            **kwargs,
        )

    def compute_reward(
        self,
        achieved_goal: "dict[str, np.ndarray]",
        desired_goal: "dict[str, np.ndarray]",
        info: "dict[str, Any]",
    ):
        for task in info["tasks_to_complete"]:
            distance = np.linalg.norm(achieved_goal[task] - desired_goal[task])
            complete = distance < BONUS_THRESH
            if complete:
                self.step_task_completions.append(task)

        return float(len(self.step_task_completions))

    def _get_obs(self, robot_obs):
        obj_qpos = self.data.qpos[9:].copy()
        obj_qvel = self.data.qvel[9:].copy()

        # Simulate observation noise
        obj_qpos += self.object_noise_ratio * self.robot_env.np_random.uniform(
            low=-1.0, high=1.0, size=obj_qpos.shape
        )
        obj_qvel += self.object_noise_ratio * self.robot_env.np_random.uniform(
            low=-1.0, high=1.0, size=obj_qvel.shape
        )

        achieved_goal = {
            task: get_joint_qpos(self.model, self.data, task).copy()
            for task in self.goal.keys()
        }

        obs = {
            "observation": np.concatenate((robot_obs, obj_qpos, obj_qvel)),
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal,
        }

        return obs

    def step(self, action):
        robot_obs, _, terminated, truncated, info = self.robot_env.step(action)
        obs = self._get_obs(robot_obs)
        info = {"tasks_to_complete": self.tasks_to_complete}

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        if self.remove_task_when_completed:
            # When the task is accomplished remove from the list of tasks to be completed
            [
                self.tasks_to_complete.remove(element)
                for element in self.step_task_completions
            ]

        info["step_task_completions"] = self.step_task_completions
        self.episode_task_completions += self.step_task_completions
        info["episode_task_completions"] = self.episode_task_completions
        self.step_task_completions.clear()
        if self.terminate_on_tasks_completed:
            # terminate if there are no more tasks to complete
            terminated = len(self.episode_task_completions) == len(self.goal.keys())

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.episode_task_completions.clear()
        robot_obs, _ = self.robot_env.reset(seed=seed)
        obs = self._get_obs(robot_obs)
        self.task_to_complete = self.goal.copy()

        info = {
            "tasks_to_complete": self.task_to_complete,
            "episode_task_completions": [],
            "step_task_completions": [],
        }

        return obs, info

    def render(self):
        return self.robot_env.render()

    def close(self):
        self.robot_env.close()
