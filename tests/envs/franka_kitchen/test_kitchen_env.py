from copy import deepcopy

import gymnasium as gym
import pytest

from gymnasium_robotics.envs.franka_kitchen.kitchen_env import (
    OBS_ELEMENT_GOALS,
    OBS_ELEMENT_INDICES,
)

TASKS = ["microwave", "kettle"]


@pytest.mark.parametrize(
    "remove_task_when_completed, terminate_on_tasks_completed",
    [[True, True], [False, False]],
)
def test_task_completion(remove_task_when_completed, terminate_on_tasks_completed):
    """This test checks the different task completion configurations for the FrankaKitchen-v1 environment.

    The test checks if the info items returned in each step (`tasks_to_complete`, `step_task_completions`, `episode_task_completions`) are correct and correspond
    to the behavior of the environment configured at initialization with the arguments: `remove_task_when_completed` and `terminate_on_tasks_completed`.
    """
    env = gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=TASKS,
        remove_task_when_completed=remove_task_when_completed,
        terminate_on_tasks_completed=terminate_on_tasks_completed,
    )
    # Test task completion for 3 consecutive episodes
    for _ in range(3):
        tasks_to_complete = deepcopy(TASKS)
        completed_tasks = set()
        _, info = env.reset()
        assert set(info["tasks_to_complete"]) == set(
            TASKS
        ), f"The item `tasks_to_complete` returned by info when the environment is reset: {set(info['tasks_to_complete'])}, must be equal to the `task_to_complete` argument used to initialize the environment: {tasks_to_complete}."
        assert (
            len(info["step_task_completions"]) == 0
        ), f"The key `step_task_completions` returned by info when the environment is reset: {set(info['step_task_completions'])}, must be empty."
        assert (
            len(info["episode_task_completions"]) == 0
        ), f"The key `episode_task_completions` returned by info when the environment is reset: {set(info['episode_task_completions'])}, must be empty."

        terminated = False

        # Complete a task sequentially for each environment step
        for task in TASKS:
            # Force task to be achieved
            env.data.qpos[OBS_ELEMENT_INDICES[task]] = OBS_ELEMENT_GOALS[task]
            _, _, terminated, _, info = env.step(env.action_space.sample())
            completed_tasks.add(task)

            assert (
                set(info["episode_task_completions"]) == completed_tasks
            ), f"The key `episode_task_completions` returned by info: {set(info['episode_task_completions'])}, must be equal to the tasks along the current episode: {completed_tasks}."
            if remove_task_when_completed:
                tasks_to_complete.remove(task)
                assert set(info["tasks_to_complete"]) == set(
                    tasks_to_complete
                ), f"If environment is initialized with `remove_task_when_completed=True` the item `tasks_to_complete` returned by info: {set(info['tasks_to_complete'])}, must be equal to the tasks that haven't been completed yet: {tasks_to_complete}."
                assert set(info["step_task_completions"]) == {
                    task
                }, f"The key `step_task_completions` returned by info: {set(info['step_task_completions'])}, must be equal to the tasks completed after the current step: {task}."

            else:
                assert set(info["tasks_to_complete"]) == set(
                    tasks_to_complete
                ), f"If environment is initialized with `remove_task_when_completed=False` the item `tasks_to_complete` returned by info: {set(info['tasks_to_complete'])}, must be equal to the set of tasks the environment was initialized with: {tasks_to_complete}."
                assert (
                    set(info["step_task_completions"]) == completed_tasks
                ), f"The key `step_task_completions` returned by info: {set(info['step_task_completions'])}, must be equal to the tasks completed after the current step: {completed_tasks}."

        if terminate_on_tasks_completed:
            assert (
                terminated
            ), "If the environment is initialized with `terminate_on_tasks_complete=True`, the episode must terminate after all tasks are completed."
        else:
            assert (
                not terminated
            ), "If the environment is initialized with `terminate_on_tasks_complete=False`, the episode must not terminate after all tasks are completed."

    # Complete a task during the same environment step
    for _ in range(3):
        tasks_to_complete = deepcopy(TASKS)
        completed_tasks = set()
        _, info = env.reset()

        terminated = False

        # Complete a task sequentially for each environment step
        for task in TASKS:
            # Force task to be achieved
            env.data.qpos[OBS_ELEMENT_INDICES[task]] = OBS_ELEMENT_GOALS[task]
            completed_tasks.add(task)

        _, _, terminated, _, info = env.step(env.action_space.sample())
        assert (
            set(info["step_task_completions"]) == completed_tasks
        ), f"The key `step_task_completions` returned by info: {set(info['step_task_completions'])}, must be equal to the tasks completed after the current step: {completed_tasks}."
        assert (
            set(info["episode_task_completions"]) == completed_tasks
        ), f"The key `episode_task_completions` returned by info: {set(info['episode_task_completions'])}, must be equal to the tasks along the current episode: {completed_tasks}."
        if remove_task_when_completed:
            assert (
                len(info["tasks_to_complete"]) == 0
            ), f"If environment is initialized with `remove_task_when_completed=True` and all tasks were completed the item `tasks_to_complete` returned by info: {set(info['tasks_to_complete'])}, must be empty."

        else:
            assert set(info["tasks_to_complete"]) == set(
                tasks_to_complete
            ), f"If environment is initialized with `remove_task_when_completed=False` the item `tasks_to_complete` returned by info: {set(info['tasks_to_complete'])}, must be equal to the set of tasks the environment was initialized with: {tasks_to_complete}."

        if terminate_on_tasks_completed:
            assert (
                terminated
            ), "If the environment is initialized with `terminate_on_tasks_complete=True`, the episode must terminate after all tasks are completed."
        else:
            assert (
                not terminated
            ), "If the environment is initialized with `terminate_on_tasks_complete=False`, the episode must not terminate after all tasks are completed."

    env.close()
