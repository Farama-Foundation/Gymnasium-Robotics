# noqa: D104
from gymnasium.envs.registration import register

from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.maze import maps
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v0


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

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
            entry_point="gymnasium_robotics.envs.fetch.slide:MujocoPyFetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchSlide{suffix}-v2",
            entry_point="gymnasium_robotics.envs.fetch.slide:MujocoFetchSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v1",
            entry_point="gymnasium_robotics.envs.fetch.pick_and_place:MujocoPyFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPickAndPlace{suffix}-v2",
            entry_point="gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v1",
            entry_point="gymnasium_robotics.envs.fetch.reach:MujocoPyFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchReach{suffix}-v2",
            entry_point="gymnasium_robotics.envs.fetch.reach:MujocoFetchReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v1",
            entry_point="gymnasium_robotics.envs.fetch.push:MujocoPyFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"FetchPush{suffix}-v2",
            entry_point="gymnasium_robotics.envs.fetch.push:MujocoFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        # Hand
        register(
            id=f"HandReach{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.reach:MujocoPyHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandReach{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.reach:MujocoHandReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "z",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateZ_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateZ_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
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
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "parallel",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateParallel_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateParallel_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateParallel_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateParallel_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
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
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateXYZ_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateXYZ_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlockRotateXYZ_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
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
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlockFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateBlock{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoPyHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block:MujocoHandBlockEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateBlock_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlock_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlock_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoPyHandBlockTouchSensorsEnv",
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
            id=f"HandManipulateBlock_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_block_touch_sensors:MujocoHandBlockTouchSensorsEnv",
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
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggRotate_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
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
            id=f"HandManipulateEggRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
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
            id=f"HandManipulateEggRotate_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
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
            id=f"HandManipulateEggRotate_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
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
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEggFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        # Alias for "Full"
        register(
            id=f"HandManipulateEgg{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoPyHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg:MujocoHandEggEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulateEgg_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
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
            id=f"HandManipulateEgg_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
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
            id=f"HandManipulateEgg_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoPyHandEggTouchSensorsEnv",
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
            id=f"HandManipulateEgg_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_egg_touch_sensors:MujocoHandEggTouchSensorsEnv",
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
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "ignore",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenRotate_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
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
            id=f"HandManipulatePenRotate_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
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
            id=f"HandManipulatePenRotate_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
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
            id=f"HandManipulatePenRotate_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
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
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePenFull{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoPyHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen:MujocoHandPenEnv",
            kwargs=_merge(
                {
                    "target_position": "random",
                    "target_rotation": "xyz",
                },
                kwargs,
            ),
            max_episode_steps=100,
        )

        register(
            id=f"HandManipulatePen_BooleanTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
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
            id=f"HandManipulatePen_BooleanTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
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
            id=f"HandManipulatePen_ContinuousTouchSensors{suffix}-v0",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoPyHandPenTouchSensorsEnv",
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
            id=f"HandManipulatePen_ContinuousTouchSensors{suffix}-v1",
            entry_point="gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_pen_touch_sensors:MujocoHandPenTouchSensorsEnv",
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

        #####################
        # D4RL Environments #
        #####################

        # ----- AntMaze -----

        register(
            id=f"AntMaze_UMaze{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_UMaze{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_G{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Open_Diverse_GR{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=700,
        )

        register(
            id=f"AntMaze_Medium{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_G{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Medium_Diverse_GR{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_G{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.ant_maze:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        register(
            id=f"AntMaze_Large_Diverse_GR{suffix}-v4",
            entry_point="gymnasium_robotics.envs.maze.ant_maze_v4:AntMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=1000,
        )

        # ----- PointMaze -----

        register(
            id=f"PointMaze_UMaze{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.U_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Open_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.OPEN_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=300,
        )

        register(
            id=f"PointMaze_Medium{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Medium_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Medium_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=600,
        )

        register(
            id=f"PointMaze_Large{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        register(
            id=f"PointMaze_Large_Diverse_G{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_G,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

        register(
            id=f"PointMaze_Large_Diverse_GR{suffix}-v3",
            entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
            kwargs=_merge(
                {
                    "maze_map": maps.LARGE_MAZE_DIVERSE_GR,
                },
                kwargs,
            ),
            max_episode_steps=800,
        )

    for reward_type in ["sparse", "dense"]:
        suffix = "Sparse" if reward_type == "sparse" else ""
        version = "v1"
        kwargs = {
            "reward_type": reward_type,
        }

        register(
            id=f"AdroitHandDoor{suffix}-{version}",
            entry_point="gymnasium_robotics.envs.adroit_hand.adroit_door:AdroitHandDoorEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandHammer{suffix}-{version}",
            entry_point="gymnasium_robotics.envs.adroit_hand.adroit_hammer:AdroitHandHammerEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandPen{suffix}-{version}",
            entry_point="gymnasium_robotics.envs.adroit_hand.adroit_pen:AdroitHandPenEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

        register(
            id=f"AdroitHandRelocate{suffix}-{version}",
            entry_point="gymnasium_robotics.envs.adroit_hand.adroit_relocate:AdroitHandRelocateEnv",
            max_episode_steps=200,
            kwargs=kwargs,
        )

    register(
        id="FrankaKitchen-v1",
        entry_point="gymnasium_robotics.envs.franka_kitchen:KitchenEnv",
        max_episode_steps=280,
    )


__version__ = "1.2.4"


try:
    import sys

    from farama_notifications import notifications

    if (
        "gymnasium_robotics" in notifications
        and __version__ in notifications["gymnasium_robotics"]
    ):
        print(notifications["gymnasium_robotics"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass
