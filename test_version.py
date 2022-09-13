import gym

all_testing_env_specs = [
    env_spec
    for env_spec in gym.envs.registry.values()
    if env_spec.entry_point.startswith("gym_robotics.envs")
]
non_mujoco_py_env_specs = [
    spec for spec in all_testing_env_specs if "MujocoPy" not in spec.entry_point
]

print(len(all_testing_env_specs))
print(len(non_mujoco_py_env_specs))
