import pickle

import gymnasium as gym

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


def test_serialize_deserialize():
    env1 = gym.make("HandReach-v2", distance_threshold=1e-6)
    env1.reset()
    env2 = pickle.loads(pickle.dumps(env1))

    assert env1.unwrapped.distance_threshold == env2.unwrapped.distance_threshold, (
        env1.distance_threshold,
        env2.distance_threshold,
    )
