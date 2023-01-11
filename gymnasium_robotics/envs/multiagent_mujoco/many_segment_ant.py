"""File for ManySegmentAntEnv.

This file is originally from the `schroederdewitt/multiagent_mujoco` repository hosted on GitHub
(https://github.com/schroederdewitt/multiagent_mujoco/blob/master/multiagent_mujoco/manyagent_ant.py)
Original Author: Schroeder de Witt

 - General code cleanup, factorization, type hinting, adding documentation and comments
 - updated API to Gymnasium.MuJoCo v4
 - increase returned info
 - renamed ManyAgentAntEnv -> ManySegmentAntEnv (and changed the __init__ arguments accordingly)
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


class ManySegmentAntEnv(mujoco_env.MujocoEnv, EzPickle):
    """Is a vartion of the Ant environment, but with ants coupled together (each segment has a torso + 4 legs).

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
            n_segs: the number of segments of the ant (1 segment is the same as Gymansium's ant)
            render_mode: see [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
        """
        self.healthy_reward = 1
        self._ctrl_cost_weight = 0.5
        self._contact_cost_weight = 5e-4
        self._reset_noise_scale = 0.1

        # Check whether asset file exists already, otherwise create it
        asset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            f"many_segment_ant_{n_segs}_segments.auto.xml",
        )
        self._generate_asset(n_segs=n_segs, asset_path=asset_path)

        observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_segs * 50 + 17,),
            dtype=np.float64,
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
            "many_segment_ant.xml.template",
        )
        with open(template_path) as file:
            template = Template(file.read())
        body_str_template = """
        <body name="torso_{:d}" pos="-1 0 0">
           <!--<joint axis="0 1 0" name="nnn_{:d}" pos="0.0 0.0 0.0" range="-1 1" type="hinge"/>-->
            <geom density="100" fromto="1 0 0 0 0 0" size="0.1" type="capsule"/>
            <body name="front_right_leg_{:d}" pos="0 0 0">
              <geom fromto="0.0 0.0 0.0 0.0 0.2 0.0" name="aux1_geom_{:d}" size="0.08" type="capsule"/>
              <body name="aux_2_{:d}" pos="0.0 0.2 0">
                <joint axis="0 0 1" name="hip1_{:d}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -0.2 0.2 0.0" name="right_leg_geom_{:d}" size="0.08" type="capsule"/>
                <body pos="-0.2 0.2 0">
                  <joint axis="1 1 0" name="ankle1_{:d}" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 0.4 0.0" name="right_ankle_geom_{:d}" size="0.08" type="capsule"/>
                </body>
              </body>
            </body>
            <body name="back_leg_{:d}" pos="0 0 0">
              <geom fromto="0.0 0.0 0.0 0.0 -0.2 0.0" name="aux2_geom_{:d}" size="0.08" type="capsule"/>
              <body name="aux2_{:d}" pos="0.0 -0.2 0">
                <joint axis="0 0 1" name="hip2_{:d}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                <geom fromto="0.0 0.0 0.0 -0.2 -0.2 0.0" name="back_leg_geom_{:d}" size="0.08" type="capsule"/>
                <body pos="-0.2 -0.2 0">
                  <joint axis="-1 1 0" name="ankle2_{:d}" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
                  <geom fromto="0.0 0.0 0.0 -0.4 -0.4 0.0" name="third_ankle_geom_{:d}" size="0.08" type="capsule"/>
                </body>
              </body>
            </body>
        """

        body_close_str_template = "</body>\n"
        actuator_str_template = """\t     <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip1_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle1_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip2_{:d}" gear="150"/>
                                          <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle2_{:d}" gear="150"/>\n"""

        body_str = ""
        for i in range(1, n_segs):
            body_str += body_str_template.format(*([i] * 16))
        body_str += body_close_str_template * (n_segs - 1)

        actuator_str = ""
        for i in range(n_segs):
            actuator_str += actuator_str_template.format(*([i] * 8))

        rt = template.render(body=body_str, actuators=actuator_str)
        with open(asset_path, "w") as file:
            file.write(rt)

    def step(self, action: np.ndarray):
        """Performs a single step given the `action`.

        Reward has same structure as ant
        Does terminate when the front torso is unhealthy (in the same way as the ant)
        """
        x_position_before = self.get_body_com("torso_0")[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.get_body_com("torso_0")[0]

        x_velocity = (x_position_after - x_position_before) / self.dt

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        ctrl_cost = self._ctrl_cost_weight * np.square(action).sum()
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1))
        )
        contact_cost = 0  # In Gymnasium.MuJoCo-v4 contanct costs are ignored

        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost - contact_cost + healthy_reward
        terminated = not notdone
        truncated = False

        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": x_position_after,
            # "y_position": xy_position_after[1],
            # "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            # "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return (observation, reward, terminated, truncated, info)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos.flat[2:],
                self.data.qvel.flat,
                np.clip(self.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self) -> np.ndarray:
        """Resets the model in same way as the Swimmer."""
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq,
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
        )
        qvel = (
            self.init_qvel
            + self.np_random.standard_normal(self.model.nv) * self._reset_noise_scale
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
