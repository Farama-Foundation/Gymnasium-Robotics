"""See the [Doc Page](https://robotics.farama.org/envs/fetch/MaMuJoCo)."""

import multiagent_mujoco.mamujoco_v0  # noqa: F401

from .coupled_half_cheetah import CoupledHalfCheetah  # noqa: F401
from .many_segment_ant import ManySegmentAntEnv  # noqa: F401
from .many_segment_swimmer import ManySegmentSwimmerEnv  # noqa: F401
from .mujoco_multi import MultiAgentMujocoEnv  # noqa: F401
