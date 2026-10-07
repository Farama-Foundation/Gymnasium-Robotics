import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

BINARY_ENV_IDS = [
    "AdroitHandDoorBinary-v1",
    "AdroitHandHammerBinary-v1",
    "AdroitHandPenBinary-v1",
    "AdroitHandRelocateBinary-v1",
]

EXPECTED_HORIZONS = {
    "AdroitHandDoorBinary-v1": 200,
    "AdroitHandHammerBinary-v1": 200,
    "AdroitHandPenBinary-v1": 100,
    "AdroitHandRelocateBinary-v1": 200,
}

DENSE_ENV_IDS = [
    "AdroitHandDoor-v2",
    "AdroitHandHammer-v2",
    "AdroitHandPen-v2",
    "AdroitHandRelocate-v2",
]


@pytest.mark.parametrize("env_id", BINARY_ENV_IDS)
def test_binary_envs_are_registered_with_the_binary_reward_type(env_id):
    env = gym.make(env_id, disable_env_checker=True)
    try:
        assert env.unwrapped.reward_type == "binary"
        # The binary variant is its own reward type, not the sparse one.
        assert env.unwrapped.sparse_reward is False
    finally:
        env.close()


@pytest.mark.parametrize("env_id", BINARY_ENV_IDS)
def test_binary_envs_only_emit_zero_or_minus_one(env_id):
    env = gym.make(env_id, disable_env_checker=True)
    try:
        env.reset(seed=0)
        for _ in range(25):
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert reward in (0.0, -1.0), reward
            # 0 marks success, -1 a per-step cost until the task is solved.
            assert reward == float(info["success"]) - 1.0
            if terminated or truncated:
                env.reset()
    finally:
        env.close()


def test_binary_reward_is_zero_on_success():
    # A random policy never opens the door, so drive the hinge past its success
    # threshold directly and check the reward flips to 0 through the env's own
    # success test rather than a stubbed one.
    env = gym.make("AdroitHandDoorBinary-v1", disable_env_checker=True)
    try:
        unwrapped = env.unwrapped
        unwrapped.reset(seed=0)
        qpos = unwrapped.data.qpos.copy()
        qvel = unwrapped.data.qvel.copy()
        qpos[unwrapped.door_hinge_addrs] = 1.6  # success is hinge >= 1.35
        unwrapped.set_state(qpos, qvel)

        _, reward, _, _, info = unwrapped.step(np.zeros(unwrapped.action_space.shape))
        assert bool(info["success"]) is True
        assert reward == 0.0
    finally:
        env.close()


@pytest.mark.parametrize("env_id", BINARY_ENV_IDS)
def test_binary_envs_use_the_d4rl_horizons(env_id):
    assert gym.spec(env_id).max_episode_steps == EXPECTED_HORIZONS[env_id]


@pytest.mark.parametrize("env_id", BINARY_ENV_IDS)
def test_reward_type_survives_a_pickle_round_trip(env_id):
    # EzPickle has to carry reward_type, or an unpickled env silently falls back
    # to the default dense rewards.
    import pickle

    env = gym.make(env_id, disable_env_checker=True)
    try:
        env.reset(seed=0)
        restored = pickle.loads(pickle.dumps(env.unwrapped))
        try:
            assert restored.reward_type == "binary"
        finally:
            restored.close()
    finally:
        env.close()


@pytest.mark.parametrize("env_id", DENSE_ENV_IDS)
def test_dense_variants_are_unchanged(env_id):
    # The new reward type is additive: the existing ids keep dense rewards.
    env = gym.make(env_id, disable_env_checker=True)
    try:
        assert env.unwrapped.reward_type == "dense"
        assert env.unwrapped.sparse_reward is False
        env.reset(seed=0)
        rewards = [env.step(env.action_space.sample())[1] for _ in range(10)]
        # Dense rewards are continuous, so they cannot all be 0/-1.
        assert not set(np.unique(rewards)).issubset({0.0, -1.0})
    finally:
        env.close()


@pytest.mark.parametrize(
    "entry_point",
    [
        "gymnasium_robotics.envs.adroit_hand.adroit_door:AdroitHandDoorEnv",
        "gymnasium_robotics.envs.adroit_hand.adroit_hammer:AdroitHandHammerEnv",
        "gymnasium_robotics.envs.adroit_hand.adroit_pen:AdroitHandPenEnv",
        "gymnasium_robotics.envs.adroit_hand.adroit_relocate:AdroitHandRelocateEnv",
    ],
)
def test_unknown_reward_type_is_rejected(entry_point):
    module_name, class_name = entry_point.split(":")
    import importlib

    env_class = getattr(importlib.import_module(module_name), class_name)
    with pytest.raises(ValueError, match="Unknown reward type"):
        env_class(reward_type="not-a-reward-type")
