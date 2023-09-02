import gymnasium as gym
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS


def test_task_completion():

    # terminate_on_tasks_completed=True and remove_task_when_completed=True
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'])
    for _ in range(5):
        _, info = env.reset()
        assert set(info['tasks_to_complete']) == {'microwave', 'kettle'}
        assert len(info['step_task_completions']) == 0, ""
        assert len(info['episode_task_completions']) == 0

        # force achieved kettle goal
        env.data.qpos[OBS_ELEMENT_INDICES["kettle"]] = OBS_ELEMENT_GOALS["kettle"]
        _, _, _, _, info = env.step(env.action_space.sample())
        assert set(info['tasks_to_complete']) == {'microwave'}
        assert set(info['step_task_completions']) == {'kettle'}
        assert set(info['episode_task_completions']) == {'kettle'}
        
        # force achieved microwave goal
        env.data.qpos[OBS_ELEMENT_INDICES["microwave"]] = OBS_ELEMENT_GOALS["microwave"]
        _, _, terminated, _, info = env.step(env.action_space.sample())
        assert len(info['tasks_to_complete']) == 0
        assert set(info['step_task_completions']) == {'microwave'}
        assert set(info['episode_task_completions']) == {'microwave'}
        assert terminated

    env.close
# `tasks_to_complete` (list[str]): list of tasks that haven't yet been completed in the current episode.
#     - `step_task_completions` (list[str]): list of tasks completed in the step taken.
#     - `episode_task_completions` (list[str]): list of tasks completed during the episode uptil the current step.