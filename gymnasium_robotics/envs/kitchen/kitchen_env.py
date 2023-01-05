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

        self.tasks_to_complete = set(tasks_to_complete)
        self.goal = {task: OBS_ELEMENT_GOALS[task] for task in tasks_to_complete}

        # Tasks completed in the current environment step
        self.step_task_completions = []
        self.object_noise_ratio = object_noise_ratio

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
        self.step_task_completions.clear()

        if self.terminate_on_tasks_completed:
            # terminate if there are no more tasks to complete
            terminated = not self.tasks_to_complete

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        robot_obs, _ = self.robot_env.reset(seed=seed)
        obs = self._get_obs(robot_obs)
        self.task_to_complete = self.goal.copy()

        info = {"tasks_to_complete": self.task_to_complete, "step_task_completions": []}

        return obs, info

    def render(self):
        return self.robot_env.render()
