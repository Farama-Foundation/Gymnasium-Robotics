import gym

env = gym.make("FetchReach-v3")

env.reset()

while True:
    action = env.action_space.sample()

    obs, rew, done, info = env.step(action)

    if done:
        env.reset()
