import gymnasium as gym
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


# One environment per family that overrides the goal-env methods.
@pytest.mark.parametrize(
    "env_id",
    [
        "FetchReach-v4",
        "HandReach-v3",
        "HandManipulateBlock-v1",
        "PointMaze_UMaze-v3",
        "AntMaze_UMaze-v3",
    ],
)
@pytest.mark.parametrize(
    "method", ["compute_reward", "compute_terminated", "compute_truncated"]
)
def test_goal_methods_take_the_documented_keywords(env_id: str, method: str):
    """`GoalEnv` names the arguments `achieved_goal` and `desired_goal`.

    Overrides that renamed them broke every keyword call: the Fetch and hand
    environments called the second one `goal`, and `BaseRobotEnv.compute_truncated`
    carried a typo, `achievec_goal`. `core.GoalEnv` documents the interface these
    override, so the names have to match it.
    """
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)

    getattr(env.unwrapped, method)(
        achieved_goal=obs["achieved_goal"],
        desired_goal=obs["desired_goal"],
        info={},
    )

    env.close()
