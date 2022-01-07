from gym.envs.registration import register

from gym_robotics.core import GoalEnv


def register_robotics_envs():
    def _merge(a, b):
        a.update(b)
        return a

    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        # Fetch
        register(
            id=f"FetchSlide{suffix}-v1",
            entry_point="gym_robotics.envs:FetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v1",
            entry_point="gym_robotics.envs:FetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v1",
            entry_point="gym_robotics.envs:FetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v1",
            entry_point="gym_robotics.envs:FetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        # Hand
        register(
            id=f"HandReach{suffix}-v0",
            entry_point="gym_robotics.envs:HandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockEnv",
            kwargs=_merge(
                {"target_position": "ignore", "target_rotation": "z"}, kwargs
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZTouchSensors{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZTouchSensors{suffix}-v1",
            entry_point="gym_robotics.envs:HandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockEnv",
            kwargs=_merge(
                {"target_position": "ignore", "target_rotation": "parallel"}, kwargs
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallelTouchSensors{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallelTouchSensors{suffix}-v1",
            entry_point="gym_robotics.envs:HandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockEnv",
            kwargs=_merge(
                {"target_position": "ignore", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZTouchSensors{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZTouchSensors{suffix}-v1",
            entry_point="gym_robotics.envs:HandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockFull{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockEnv",
            kwargs=_merge(
                {"target_position": "random", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateBlock{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockEnv",
            kwargs=_merge(
                {"target_position": "random", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockTouchSensors{suffix}-v0",
            entry_point="gym_robotics.envs:HandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockTouchSensors{suffix}-v1",
            entry_point="gym_robotics.envs:HandBlockTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate{suffix}-v0",
            entry_point="gym_robotics.envs:HandEggEnv",
            kwargs=_merge(
                {"target_position": "ignore", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotateTouchSensors{suffix}-v0",
            entry_point="gym_robotics.envs:HandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotateTouchSensors{suffix}-v1",
            entry_point="gym_robotics.envs:HandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggFull{suffix}-v0",
            entry_point="gym_robotics.envs:HandEggEnv",
            kwargs=_merge(
                {"target_position": "random", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateEgg{suffix}-v0",
            entry_point="gym_robotics.envs:HandEggEnv",
            kwargs=_merge(
                {"target_position": "random", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggTouchSensors{suffix}-v0",
            entry_point="gym_robotics.envs:HandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggTouchSensors{suffix}-v1",
            entry_point="gym_robotics.envs:HandEggTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate{suffix}-v0",
            entry_point="gym_robotics.envs:HandPenEnv",
            kwargs=_merge(
                {"target_position": "ignore", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotateTouchSensors{suffix}-v0",
            entry_point="gym_robotics.envs:HandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotateTouchSensors{suffix}-v1",
            entry_point="gym_robotics.envs:HandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenFull{suffix}-v0",
            entry_point="gym_robotics.envs:HandPenEnv",
            kwargs=_merge(
                {"target_position": "random", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulatePen{suffix}-v0",
            entry_point="gym_robotics.envs:HandPenEnv",
            kwargs=_merge(
                {"target_position": "random", "target_rotation": "xyz"}, kwargs
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenTouchSensors{suffix}-v0",
            entry_point="gym_robotics.envs:HandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "boolean",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenTouchSensors{suffix}-v1",
            entry_point="gym_robotics.envs:HandPenTouchSensorsEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                    "touch_get_obs": "sensordata",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )


from . import _version

__version__ = _version.get_versions()["version"]
