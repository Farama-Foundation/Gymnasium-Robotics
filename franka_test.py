import gymnasium as gym

env = gym.make('FrankaKitchen-v1', render_mode='human')

for _ in range(1000):
    env.reset()
    while True:
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
        print(obs['observation'].shape)
        print(env.observation_space)
        print(info)
        if terminated or truncated:
            break
