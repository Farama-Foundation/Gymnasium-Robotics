import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

import gymnasium_robotics
from gymnasium_robotics.envs.mujoco.mujoco_py_env import BaseMujocoPyEnv

gym.register_envs(gymnasium_robotics)

ALL_MUJOCO_ENVS = [
    "Ant",
    "HalfCheetah",
    "Hopper",
    "Humanoid",
    "HumanoidStandup",
    "InvertedDoublePendulum",
    "InvertedPendulum",
    "Pusher",
    "Reacher",
    "Swimmer",
    "Walker2d",
]


# Note: "HumnanoidStandup-v4" does not have `info`
# Note: "Humnanoid-v4/3" & "Ant-v4/3" fail this test
@pytest.mark.parametrize(
    "env_id",
    [
        "HalfCheetah-v3",
        "Hopper-v3",
        "Swimmer-v3",
        "Walker2d-v3",
    ],
)
def test_verify_info_x_position(env_id: str):
    """Asserts that the environment has position[0] == info['x_position']."""
    env = gym.make(env_id, exclude_current_positions_from_observation=False)

    _, _ = env.reset()
    obs, _, _, _, info = env.step(env.action_space.sample())

    assert obs[0] == info["x_position"]


# Note: "HumnanoidStandup-v4" does not have `info`
# Note: "Humnanoid-v4/3" & "Ant-v4/3" fail this test
@pytest.mark.parametrize(
    "env_id",
    [
        "Swimmer-v3",
    ],
)
def test_verify_info_y_position(env_id: str):
    """Asserts that the environment has position[1] == info['y_position']."""
    env = gym.make(env_id, exclude_current_positions_from_observation=False)

    _, _ = env.reset()
    obs, _, _, _, info = env.step(env.action_space.sample())

    assert obs[1] == info["y_position"]


# Note: "HumnanoidStandup-v4" does not have `info`
@pytest.mark.parametrize("env_name", ["HalfCheetah", "Hopper", "Swimmer", "Walker2d"])
@pytest.mark.parametrize("version", ["v3"])
def test_verify_info_x_velocity(env_name: str, version: str):
    """Asserts that the environment `info['x_velocity']` is properly assigned."""
    env = gym.make(f"{env_name}-{version}").unwrapped
    assert isinstance(env, (MujocoEnv, BaseMujocoPyEnv))
    env.reset()

    old_x = env.data.qpos[0]
    _, _, _, _, info = env.step(env.action_space.sample())
    new_x = env.data.qpos[0]

    dx = new_x - old_x
    vel_x = dx / env.dt
    assert vel_x == info["x_velocity"]


# Note: "HumnanoidStandup-v4" does not have `info`
@pytest.mark.parametrize("env_id", ["Swimmer-v3"])
def test_verify_info_y_velocity(env_id: str):
    """Asserts that the environment `info['y_velocity']` is properly assigned."""
    env = gym.make(env_id).unwrapped
    assert isinstance(env, (MujocoEnv, BaseMujocoPyEnv))
    env.reset()

    old_y = env.data.qpos[1]
    _, _, _, _, info = env.step(env.action_space.sample())
    new_y = env.data.qpos[1]

    dy = new_y - old_y
    vel_y = dy / env.dt
    assert vel_y == info["y_velocity"]


@pytest.mark.parametrize("env_id", ["Ant-v3"])
def test_verify_info_xy_velocity_xpos(env_id: str):
    """Asserts that the environment `info['x/y_velocity']` is properly assigned, for the ant environment which uses kinmatics for the velocity."""
    env = gym.make(env_id).unwrapped
    assert isinstance(env, (MujocoEnv, BaseMujocoPyEnv))
    env.reset()

    old_xy = env.get_body_com("torso")[:2].copy()
    _, _, _, _, info = env.step(env.action_space.sample())
    new_xy = env.get_body_com("torso")[:2].copy()

    dxy = new_xy - old_xy
    vel_x, vel_y = dxy / env.dt
    assert vel_x == info["x_velocity"]
    assert vel_y == info["y_velocity"]


@pytest.mark.parametrize("env_id", ["Humanoid-v3"])
def test_verify_info_xy_velocity_com(env_id: str):
    """Asserts that the environment `info['x/y_velocity']` is properly assigned, for the humanoid environment which uses kinmatics of Center Of Mass for the velocity."""

    def mass_center(model, data):
        mass = np.expand_dims(model.body_mass, axis=1)
        xpos = data.xipos
        return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()

    env = gym.make(env_id).unwrapped
    assert isinstance(env, (MujocoEnv, BaseMujocoPyEnv))
    env.reset()

    old_xy = mass_center(env.model, env.data)
    _, _, _, _, info = env.step(env.action_space.sample())
    new_xy = mass_center(env.model, env.data)

    dxy = new_xy - old_xy
    vel_x, vel_y = dxy / env.dt
    assert vel_x == info["x_velocity"]
    assert vel_y == info["y_velocity"]


@pytest.mark.parametrize("version", ["v3", "v2"])
def test_set_state(version: str):
    """Simple Test to verify that `mujocoEnv.set_state()` works correctly."""
    env = gym.make(f"Hopper-{version}").unwrapped
    assert isinstance(env, (MujocoEnv, BaseMujocoPyEnv))
    env.reset()
    new_qpos = np.array(
        [0.00136962, 1.24769787, -0.00459026, -0.00483472, 0.0031327, 0.00412756]
    )
    new_qvel = np.array(
        [0.00106636, 0.00229497, 0.00043625, 0.00435072, 0.00315854, -0.00497261]
    )
    env.set_state(new_qpos, new_qvel)
    assert (env.data.qpos == new_qpos).all()
    assert (env.data.qvel == new_qvel).all()


# Note: HumanoidStandup-v4/v3 does not have `info`
# Note: Ant-v4/v3 fails this test
# Note: Humanoid-v4/v3 fails this test
# Note: v2 does not have `info`
@pytest.mark.parametrize("env_id", ["Swimmer-v3"])
def test_distance_from_origin_info(env_id: str):
    """Verify that `info"distance_from_origin"` is correct."""
    env = gym.make(env_id).unwrapped
    assert isinstance(env, (MujocoEnv, BaseMujocoPyEnv))
    env.reset()

    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["distance_from_origin"] == np.linalg.norm(
        env.data.qpos[0:2] - env.init_qpos[0:2]
    )


# note: fails with `mujoco-mjx==3.0.1`
@pytest.mark.parametrize("version", ["v3", "v2"])
def test_model_sensors(version: str):
    """Verify that all the sensors of the model are loaded."""
    env = gym.make(f"Ant-{version}").unwrapped
    assert env.data.cfrc_ext.shape == (14, 6)

    env = gym.make(f"Humanoid-{version}").unwrapped
    assert env.data.cinert.shape == (14, 10)
    assert env.data.cvel.shape == (14, 6)
    assert env.data.qfrc_actuator.shape == (23,)
    assert env.data.cfrc_ext.shape == (14, 6)

    if version != "v3":  # HumanoidStandup v3 does not exist
        env = gym.make(f"HumanoidStandup-{version}").unwrapped
        assert env.data.cinert.shape == (14, 10)
        assert env.data.cvel.shape == (14, 6)
        assert env.data.qfrc_actuator.shape == (23,)
        assert env.data.cfrc_ext.shape == (14, 6)


@pytest.mark.parametrize(
    "env_id",
    [
        "Ant-v3",
        "HalfCheetah-v3",
        "Hopper-v3",
        "Humanoid-v3",
        "Swimmer-v3",
        "Walker2d-v3",
    ],
)
def test_reset_noise_scale(env_id):
    """Checks that when `reset_noise_scale=0` we have deterministic initialization."""
    env = gym.make(env_id, reset_noise_scale=0).unwrapped
    env.reset()

    assert np.all(env.data.qpos == env.init_qpos)
    assert np.all(env.data.qvel == env.init_qvel)
