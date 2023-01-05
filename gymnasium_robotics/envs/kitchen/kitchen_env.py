from typing import Any, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.kitchen.franka_env import FrankaRobot
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
    | --- | ----------------------------------------------------| ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | `robot:joint1` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator1                        | hinge | angle (rad) |
    | 1   | `robot:joint2` angular position                     | -1          | 1           | -1.76 (rad)  | 1.76 (rad)  | actuator2                        | hinge | angle (rad) |
    | 2   | `robot:joint3` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator3                        | hinge | angle (rad) |
    | 3   | `robot:joint4` angular position                     | -1          | 1           | -3.07 (rad)  | 0.0 (rad)   | actuator4                        | hinge | angle (rad) |
    | 4   | `robot:joint5` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator5                        | hinge | angle (rad) |
    | 5   | `robot:joint6` angular position                     | -1          | 1           | -0.0 (rad)   | 3.75 (rad)  | actuator6                        | hinge | angle (rad) |
    | 6   | `robot:joint7` angular position                     | -1          | 1           | -2.9 (rad)   | 2.9 (rad)   | actuator7                        | hinge | angle (rad) |
    | 7   | `robot:finger_left` and `robot:finger_right` force  | -1          | 1           | 0 (N)        | 255 (N)     | actuator8                        | slide | force (N)   |

    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's end effector state and goal. The kinematics observations are derived from Mujoco bodies known as [sites](https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site) attached to the body of interest such as the block or the end effector.
    Only the observations from the gripper fingers are derived from joints. Also to take into account the temporal influence of the step time, velocity values are multiplied by the step time dt=number_of_sub_steps*sub_step_time. The dictionary consists of the following 3 keys:
    `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic information of the block object and gripper. The elements of the array correspond to the following:
        | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|----------------------------------------|----------|--------------------------|
        | 0   | End effector x position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 1   | End effector y position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 2   | End effector z position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 3   | Block x position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 4   | Block y position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 5   | Block z position in global coordinates                                                                                                | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 6   | Relative block x position with respect to gripper x position in globla coordinates. Equals to x<sub>gripper</sub> - x<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 7   | Relative block y position with respect to gripper y position in globla coordinates. Equals to y<sub>gripper</sub> - y<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 8   | Relative block z position with respect to gripper z position in globla coordinates. Equals to z<sub>gripper</sub> - z<sub>block</sub> | -Inf   | Inf    | object0                               |-                                       |-         | position (m)             |
        | 9   | Joint displacement of the right gripper finger                                                                                        | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | position (m)             |
        | 10  | Joint displacement of the left gripper finger                                                                                         | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | position (m)             |
        | 11  | Global x rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
        | 12  | Global y rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
        | 13  | Global z rotation of the block in a XYZ Euler frame rotation                                                                          | -Inf   | Inf    | object0                               |-                                       |-         | angle (rad)              |
        | 14  | Relative block linear velocity in x direction with respect to the gripper                                                              | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
        | 15  | Relative block linear velocity in y direction with respect to the gripper                                                              | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
        | 16  | Relative block linear velocity in z direction                                                                                         | -Inf   | Inf    | object0                               |-                                       |-         | velocity (m/s)           |
        | 17  | Block angular velocity along the x axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
        | 18  | Block angular velocity along the y axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
        | 19  | Block angular velocity along the z axis                                                                                               | -Inf   | Inf    | object0                               |-                                       |-         | angular velocity (rad/s) |
        | 20  | End effector linear velocity x direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 21  | End effector linear velocity y direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 22  | End effector linear velocity z direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 23  | Right gripper finger linear velocity                                                                                                  | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | velocity (m/s)           |
        | 24  | Left gripper finger linear velocity                                                                                                   | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | velocity (m/s)           |

    desired_goal`: this key represents the final goal to be achieved. In this environment it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates of the desired final block position `[x,y,z]`. In order for the robot to perform a pick and place trajectory, the goal position can be elevated over the table or on top of the table. The elements of the array are the following:
        | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
        | 0   | Final goal block position in the x coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |
        | 1   | Final goal block position in the y coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |
        | 2   | Final goal block position in the z coordinate                                                                                         | -Inf   | Inf    | target0                               | position (m) |

    `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER). The value is an `ndarray` with shape `(3,)`. The elements of the array are the following:
        | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) |Unit          |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|--------------|
        | 0   | Current block position in the x coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |
        | 1   | Current block position in the y coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |
        | 2   | Current block position in the z coordinate                                                                                            | -Inf   | Inf    | object0                               | position (m) |

    ## Rewards

    The environment's reward is `sparser`. The reward in each Gymnasium step is equal to the number of task completed in the given step. If not task is completed the returned reward will be zero.

    ## Starting State

    The simulation starts with all of the joint position actuators of the Franka robot set to zero. The doors of the microwave and cabinets are closed, the burners turned off, and the light switch also off. The kettle
    will be placed in the bottom left burner.

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
            model_path="../assets/kitchen_franka/kitchen_env_model.xml", **kwargs
        )

        self.robot_env.init_qpos[:7] = np.array(
            [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, 0.0]
        )
        self.model = self.robot_env.model
        self.data = self.robot_env.data

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
            "desired_goal": self.goal.copy(),
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
        self.episode_task_completions + self.step_task_completions
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
