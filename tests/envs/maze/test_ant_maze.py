from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_reset(version):
    """Check that AntMaze does not reset into a success state."""
    env = gym.make(f"AntMaze_UMaze-{version}", continuing_task=True)

    for _ in range(1000):
        obs, info = env.reset()
        assert not info["success"]
        dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        assert dist > 0.45, f"dist={dist} < 0.45"


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_temp_xml_file_lifecycle(version):
    """Check that the generated maze XML file is cleaned up when the environment closes."""
    env = gym.make(f"AntMaze_UMaze-{version}")

    try:
        unwrapped_env = env.unwrapped
        temp_xml_path = Path(unwrapped_env.tmp_xml_file_path)
        temp_dir = Path(unwrapped_env.tmp_dir.name)

        assert temp_xml_path.exists()
        assert temp_dir.exists()
        assert temp_xml_path.parent == temp_dir
    finally:
        env.close()

    assert not temp_xml_path.exists()
    assert not temp_dir.exists()
