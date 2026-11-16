"""Keep frozen mujoco-py tests on interpreters supported by NumPy 1.x."""

import sys

import gymnasium as gym
import pytest


def pytest_collection_modifyitems(items):
    """Run legacy binding tests on Python 3.11/3.12 and modern tests everywhere."""
    if sys.version_info < (3, 13):
        return

    legacy = pytest.mark.skip(reason="mujoco-py requires NumPy 1.x and Python < 3.13")
    for item in items:
        if "tests/envs/mujoco/" in item.path.as_posix():
            item.add_marker(legacy)
            continue
        for value in getattr(getattr(item, "callspec", None), "params", {}).values():
            if isinstance(value, str) and value in gym.registry:
                entry_point = gym.registry[value].entry_point
                if isinstance(entry_point, str) and (
                    "MujocoPy" in entry_point
                    or entry_point.startswith("gymnasium_robotics.envs.mujoco.")
                ):
                    item.add_marker(legacy)
