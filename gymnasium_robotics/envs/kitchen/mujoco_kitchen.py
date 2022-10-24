from gymnasium_robotics.envs.kitchen.kitchen_env import (
    KitchenMicrowaveKettleBottomBurnerLightV0,
)

env = KitchenMicrowaveKettleBottomBurnerLightV0()

env.reset()
print(env.init_qpos)

print(env.observation_space)

obs, done, rew, info = env.step(env.action_space.sample())

print(obs.shape)
