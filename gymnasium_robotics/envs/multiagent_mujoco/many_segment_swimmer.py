"""File for ManySegmentSwimmerEnv.

This file is originally from the `schroederdewitt/multiagent_mujoco` repository hosted on GitHub
(https://github.com/schroederdewitt/multiagent_mujoco/blob/master/multiagent_mujoco/manyagent_swimmer.py)
Original Author: Schroeder de Witt

 - General code cleanup, factorization, type hinting, adding documentation and comments
 - updated API to Gymnasium.MuJoCo v4
 - increase returned info
 - renamed ManyAgentSwimmerEnv -> ManySegmentSwimmerEnv (and changed the __init__ arguments accordingly)
"""


import os
import typing

import gymnasium
import numpy as np
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.utils.ezpickle import EzPickle
from jinja2 import Template

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class ManySegmentSwimmerEnv(mujoco_env.MujocoEnv, EzPickle):
    """Is a vartion of the Swimmer environment, but with many segments.

    This environment was first introduced ["FACMAC: Factored Multi-Agent Centralised Policy Gradients"](https://arxiv.org/abs/2003.06709).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, n_segs: int, render_mode: typing.Optional[str] = None):
        """Init.

        Args:
            n_segs: the number of segments of the swimmer (3 segments is the same as Gymansium's swimmer)
            render_mode: see [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
        """
        self._forward_reward_weight = 1.0
        self._ctrl_cost_weight = 1e-4
        self._reset_noise_scale = 0.1

        # Check whether asset file exists already, otherwise create it
        asset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            f"many_segment_swimmer_{n_segs}_segments.auto.xml",
        )
        self._generate_asset(n_segs=n_segs, asset_path=asset_path)

        observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_segs * 2 + 4,), dtype=np.float64
        )
        mujoco_env.MujocoEnv.__init__(
            self,
            asset_path,
            4,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
        )
        EzPickle.__init__(self, n_segs=n_segs, render_mode=render_mode)
        os.remove(asset_path)

    def _generate_asset(self, n_segs: int, asset_path: str) -> None:
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "many_segment_swimmer.xml.template",
        )
        with open(template_path) as file:
            template = Template(file.read())
        body_str_template = """
        <body name="mid{:d}" pos="-1 0 0">
          <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
          <joint axis="0 0 {:d}" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
        """

        body_end_str_template = """
        <body name="back" pos="-1 0 0">
            <geom density="1000" fromto="0 0 0 -1 0 0" size="0.1" type="capsule"/>
            <joint axis="0 0 1" limited="true" name="rot{:d}" pos="0 0 0" range="-100 100" type="hinge"/>
          </body>
        """

        body_close_str_template = "</body>\n"
        actuator_str_template = """\t <motor ctrllimited="true" ctrlrange="-1 1" gear="150.0" joint="rot{:d}"/>\n"""

        body_str = ""
        for i in range(1, n_segs - 1):
            body_str += body_str_template.format(i, (-1) ** (i + 1), i)
        body_str += body_end_str_template.format(n_segs - 1)
        body_str += body_close_str_template * (n_segs - 2)

        actuator_str = ""
        for i in range(n_segs):
            actuator_str += actuator_str_template.format(i)

        rt = template.render(body=body_str, actuators=actuator_str)
        with open(asset_path, "w") as file:
            file.write(rt)

    def step(self, action: np.ndarray):
        """Performs a single step given the `action`.

        Reward has same structure as Swimmer
        Does never terminate (like Swimmer)
        """
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]

        x_velocity = (x_position_after - x_position_before) / self.dt

        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self._ctrl_cost_weight * np.square(action).sum()

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminal = False
        truncated = False
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": x_position_after,
            # "y_position": xy_position_after[1],
            # "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            # "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminal, truncated, info

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self) -> np.ndarray:
        """Resets the model in same way as the Swimmer."""
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(
                low=-self._reset_noise_scale,
                high=self._reset_noise_scale,
                size=self.model.nq,
            ),
            self.init_qvel
            + self.np_random.uniform(
                low=-self._reset_noise_scale,
                high=self._reset_noise_scale,
                size=self.model.nv,
            ),
        )
        return self._get_obs()
