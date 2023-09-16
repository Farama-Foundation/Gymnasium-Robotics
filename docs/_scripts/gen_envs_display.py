import os
import re

import gymnasium as gym
from tqdm import tqdm

if __name__ == "__main__":
    """
    python gen_envs_display
    """
    # REWRITE: for new environments that don't include Fetch and Shadow Hand (D4RL)
    # TODO: use same format for Fetch and Shadow Hand
    # The environment entrypoints have the following standard: `gymnasium_robotics.envs.env_type.env_name:EnvName`
    all_envs = []
    for env_spec in gym.envs.registry.values():
        if isinstance(env_spec.entry_point, str):
            if (
                env_spec.entry_point.startswith("gymnasium_robotics.envs")
                and "MujocoPy" not in env_spec.entry_point
            ):
                all_envs.append(env_spec)  # Exclude Fetch and Shadow Hand environments
    filtered_envs_by_type = {}
    for env_spec in all_envs:
        env_full_name = env_spec.entry_point.split(":")

        split_entrypoint = env_full_name[0].split(".")
        if len(split_entrypoint) == 4:
            env_type = split_entrypoint[-2]
            env_name = split_entrypoint[-1]

        if len(split_entrypoint) == 3:
            env_type = split_entrypoint[-1]
            env_name = split_entrypoint[-1]

        # Remove file version from env_name
        env_name = re.sub(r"(?:_v(?P<version>\d+))", "", env_name)

        if env_type not in filtered_envs_by_type:
            filtered_envs_by_type[env_type] = [env_name]
        elif env_name not in filtered_envs_by_type[env_type]:
            filtered_envs_by_type[env_type].append(env_name)

    for env_type, env_names in tqdm(filtered_envs_by_type.items()):
        cells = []
        for env_name in env_names:
            grid_cell = f"""
                <a href="{env_name}">
                    <div class="env-grid__cell">
                        <div class="cell__image-container">
                            <img src="/_static/videos/{env_type}/{env_name}.gif">
                        </div>
                        <div class="cell__title">
                            <span>{' '.join(env_name.split('_')).title()}</span>
                        </div>
                    </div>
                </a>
            """
            cells.append(grid_cell)

        cells = "\n".join(cells)

        page = f"""
    <div class="env-grid">
        {cells}
    </div>
        """

        fp = open(
            os.path.join(
                os.path.dirname(__file__), "..", "envs", env_type, "list.html"
            ),
            "w",
            encoding="utf-8",
        )
        fp.write(page)
        fp.close()
