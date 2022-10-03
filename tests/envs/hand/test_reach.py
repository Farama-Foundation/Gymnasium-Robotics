import pickle

import gymnasium as gym


def test_serialize_deserialize():
    env1 = gym.make("HandReach-v0", distance_threshold=1e-6)
    env1.reset()
    env2 = pickle.loads(pickle.dumps(env1))

    assert env1.distance_threshold == env2.distance_threshold, (
        env1.distance_threshold,
        env2.distance_threshold,
    )
