import os

import gymnasium as gym
from PIL import Image
from tqdm import tqdm

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

# how many steps to record an env for
LENGTH = 300

# TODO: Add Fetch and Shadow Hand environments
# The environment entrypoints have the following standard: `gymnasium_robotics.envs.env_type.env_name:EnvName`
all_envs = []
for env_spec in gym.envs.registry.values():
    if isinstance(env_spec.entry_point, str):
        if (
            env_spec.entry_point.startswith("gymnasium_robotics.envs")
            and "MujocoPy" not in env_spec.entry_point
        ):
            all_envs.append(env_spec)  # Exclude Fetch and Shadow Hand environments

# Keep latest version of environments
filtered_envs_by_version = {}
for env_spec in all_envs:
    if env_spec.name not in filtered_envs_by_version:
        filtered_envs_by_version[env_spec.name] = env_spec
    elif filtered_envs_by_version[env_spec.name].version < env_spec.version:
        filtered_envs_by_version[env_spec.name] = env_spec

filtered_envs_by_name = {}
# Get one env id per env_name. Environments can have many subenvironments. For example,
# Point Maze has different subenvironments depending on the map shape
for env_spec in filtered_envs_by_version.values():
    env_full_name = env_spec.entry_point.split(":")[0]
    if env_full_name not in filtered_envs_by_name:
        filtered_envs_by_name[env_full_name] = env_spec.id


for env_full_name, env_id in tqdm(filtered_envs_by_name.items()):
    # try catch in case missing some installs
    try:
        env = gym.make(env_id, render_mode="rgb_array")

        env_split_full_name = env_full_name.split(".")
        if len(env_split_full_name) == 4:
            env_name = env_split_full_name[-1]
            env_type = env_split_full_name[-2]
        if len(env_split_full_name) == 3:
            env_name = env_type = env_split_full_name[-1]

        # obtain and save LENGTH frames worth of steps
        frames = []
        while True:
            state, info = env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated) and len(frames) <= LENGTH:

                frame = env.render()
                frames.append(Image.fromarray(frame))
                action = env.action_space.sample()
                state_next, reward, terminated, truncated, info = env.step(action)

            if len(frames) > LENGTH:
                break

        env.close()

        # make sure video doesn't already exist
        # if not os.path.exists(os.path.join(v_path, env_name + ".gif")):
        frames[0].save(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "_static",
                "videos",
                env_type,
                env_name + ".gif",
            ),
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0,
        )
        print("Saved: " + env_name)

    except BaseException as e:
        print("ERROR", e)
        continue
