"""File for CoupledHalfCheetahEnv.

This file is originally from the `schroederdewitt/multiagent_mujoco` repository hosted on GitHub
(https://github.com/schroederdewitt/multiagent_mujoco/blob/master/multiagent_mujoco/coupled_half_cheetah.py)
Original Author: Schroeder de Witt

 - General code cleanup, factorization, type hinting, adding documentation and comments
- updated API to Gymnasium.MuJoCo v4
- increase returned info
- fixed `_get_obs()` (now also returns tendon related observations)
- renamed CoupledHalfCheetah -> CoupledHalfCheetahEnv
"""
import os
import typing

import gymnasium
import numpy as np
from gymnasium.envs.mujoco import mujoco_env
from gymnasium.utils.ezpickle import EzPickle

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class CoupledHalfCheetahEnv(mujoco_env.MujocoEnv, EzPickle):
    """Class for CoupledHalfCheetah mujoco environment.

    ## Description
    This environment was first (half-)implemented in the [original_mamujoco](https://github.com/schroederdewitt/multiagent_mujoco)
    This environment consists of 2 half cheetahs coupled by an elastic tendon.

    ## Action Space
    The action space is a `Box(-1, 1, (12,), float32)`. An action represents the torques applied between *links*.

    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the back thigh rotor of the first cheetah   | -1          | 1           | bthigh0                          | hinge | torque (N m) |
    | 1   | Torque applied on the back shin rotor of the first cheetah    | -1          | 1           | bshin0                           | hinge | torque (N m) |
    | 2   | Torque applied on the back foot rotor of the first cheetah    | -1          | 1           | bfoot0                           | hinge | torque (N m) |
    | 3   | Torque applied on the front thigh rotor of the first cheetah   | -1          | 1           | fthigh0                          | hinge | torque (N m) |
    | 4   | Torque applied on the front shin rotor of the first cheetah   | -1          | 1           | fshin0                           | hinge | torque (N m) |
    | 5   | Torque applied on the front foot rotor of the first cheetah   | -1          | 1           | ffoot0                           | hinge | torque (N m) |
    | 6   | Torque applied on the back thigh rotor of the second cheetah  | -1          | 1           | bthigh1                          | hinge | torque (N m) |
    | 7   | Torque applied on the back shin rotor of the second cheetah   | -1          | 1           | bshin1                           | hinge | torque (N m) |
    | 8   | Torque applied on the back foot rotor of the second cheetah   | -1          | 1           | bfoot1                           | hinge | torque (N m) |
    | 9   | Torque applied on the front thigh rotor of the second cheetah | -1          | 1           | fthigh1                          | hinge | torque (N m) |
    | 10  | Torque applied on the front shin rotor of the second cheetah  | -1          | 1           | fshin1                           | hinge | torque (N m) |
    | 11  | Torque applied on the front foot rotor of the second cheetah  | -1          | 1           | ffoot1                           | hinge | torque (N m) |


    ## Observation Space
    Observations consist of positional values of different body parts of the cheetahs, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities, followed by the jacobian of the tendon, the length of the tendon, and it's velocity (it's derivative).

    The observation space is a `Box(-1, 1, (40,), float64)`.

    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint  | Unit                     |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ------ | ------------------------ |
    | 0   | z-coordinate of the front tip of the first cheetah         | -Inf | Inf | rootz0                           | slide  | position (m)             |
    | 1   | angle of the front tip of the first cheetah                | -Inf | Inf | rooty0                           | hinge  | angle (rad)              |
    | 2   | angle of the second rotor of the first cheetah             | -Inf | Inf | bthigh0                          | hinge  | angle (rad)              |
    | 3   | angle of the second rotor of the first cheetah             | -Inf | Inf | bshin0                           | hinge  | angle (rad)              |
    | 4   | velocity of the tip along the x-axis of the first cheetah  | -Inf | Inf | bfoot0                           | hinge  | angle (rad)              |
    | 5   | velocity of the tip along the y-axis of the first cheetah  | -Inf | Inf | fthigh0                          | hinge  | angle (rad)              |
    | 6   | angular velocity of front tip of the first cheetah         | -Inf | Inf | fshin0                           | hinge  | angle (rad)              |
    | 7   | angular velocity of second rotor of the first cheetah      | -Inf | Inf | ffoot0                           | hinge  | angle (rad)              |
    | 8   | z-coordinate of the front tip of the second cheetah        | -Inf | Inf | rootz1                           | slide  | position (m)             |
    | 9   | angle of the front tip of the second cheetah               | -Inf | Inf | rooty1                           | hinge  | angle (rad)              |
    | 10  | angle of the second rotor of the second cheetah            | -Inf | Inf | bthigh1                          | hinge  | angle (rad)              |
    | 11  | angle of the second rotor of the second cheetah            | -Inf | Inf | bshin1                           | hinge  | angle (rad)              |
    | 12  | velocity of the tip along the x-axis of the second cheetah | -Inf | Inf | bfoot1                           | hinge  | angle (rad)              |
    | 13  | velocity of the tip along the y-axis of the second cheetah | -Inf | Inf | fthigh1                          | hinge  | angle (rad)              |
    | 14  | angular velocity of front tip  of the second cheetah       | -Inf | Inf | fshin1                           | hinge  | angle (rad)              |
    | 15  | angular velocity of second rotor of the second cheetah     | -Inf | Inf | ffoot1                           | hinge  | angle (rad)              |
    | 16  | x-coordinate of the front tip of the first cheetah         | -Inf | Inf | rootx0                           | slide  | velocity (m/s)           |
    | 17  | y-coordinate of the front tip of the first cheetah         | -Inf | Inf | rootz0                           | slide  | velocity (m/s)           |
    | 18  | angle of the front tip of the first cheetah                | -Inf | Inf | rooty0                           | hinge  | angular velocity (rad/s) |
    | 19  | angle of the second rotor of the first cheetah             | -Inf | Inf | bthigh0                          | hinge  | angular velocity (rad/s) |
    | 20  | angle of the second rotor of the first cheetah             | -Inf | Inf | bshin0                           | hinge  | angular velocity (rad/s) |
    | 21  | velocity of the tip along the x-axis of the first cheetah  | -Inf | Inf | bfoot0                           | hinge  | angular velocity (rad/s) |
    | 22  | velocity of the tip along the y-axis of the first cheetah  | -Inf | Inf | fthigh0                          | hinge  | angular velocity (rad/s) |
    | 23  | angular velocity of front tip of the first cheetah         | -Inf | Inf | fshin0                           | hinge  | angular velocity (rad/s) |
    | 24  | angular velocity of second rotor of the first cheetah      | -Inf | Inf | ffoot0                           | hinge  | angular velocity (rad/s) |
    | 25  | x-coordinate of the front tip of the second cheetah        | -Inf | Inf | rootx1                           | slide  | velocity (m/s)           |
    | 26  | y-coordinate of the front tip of the second cheetah        | -Inf | Inf | rootz1                           | slide  | velocity (m/s)           |
    | 27  | angle of the front tip of the second cheetah               | -Inf | Inf | rooty1                           | hinge  | angular velocity (rad/s) |
    | 28  | angle of the second rotor of the second cheetah            | -Inf | Inf | bthigh1                          | hinge  | angular velocity (rad/s) |
    | 29  | angle of the second rotor of the second cheetah            | -Inf | Inf | bshin1                           | hinge  | angular velocity (rad/s) |
    | 30  | velocity of the tip along the x-axis of the second cheetah | -Inf | Inf | bfoot1                           | hinge  | angular velocity (rad/s) |
    | 31  | velocity of the tip along the y-axis of the second cheetah | -Inf | Inf | fthigh1                          | hinge  | angular velocity (rad/s) |
    | 32  | angular velocity of front tip of the second cheetah        | -Inf | Inf | fshin1                           | hinge  | angular velocity (rad/s) |
    | 33  | angular velocity of second rotor of the second cheetah     | -Inf | Inf | ffoot1                           | hinge  | angular velocity (rad/s) |
    | 34  | jacobian of tendon                                         | -Inf | Inf | ten_J[0] (tendon0)               | tendon | |
    | 35  | jacobian of tendon                                         | -Inf | Inf | ten_J[1] (tendon0)               | tendon | |
    | 36  | jacobian of tendon                                         | -Inf | Inf | ten_J[9] (tendon0)               | tendon | |
    | 37  | jacobian of tendon                                         | -Inf | Inf | ten_J[10] (tendon0)              | tendon | |
    | 38  | length of tendon                                           | -Inf | Inf | ten_lenght (tendon0)             | tendon | distance (m)             |
    | 39  | tendon veloocity                                           | -Inf | Inf | ten_velocity (tendon0)           | tendon | rate of expansion (m/s)  |

    ## Rewards
    The reward uses the same structure as the (single) HalfCheetah

    The reward consists of two parts:
    - *forward_reward*: A reward of moving forward which is measured
    as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
    the time between actions and is dependent on the frame_skip parameter
    (fixed to 5), where the frametime is 0.01 - making the
    default *dt = 5 * 0.01 = 0.05*. This reward would be positive if the cheetah
    runs forward (right).
    - *ctrl_cost*: A cost for penalising the cheetahs if it takes
    actions that are too large. It is measured as *`ctrl_cost_weight` *
    sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
    control and has a default value of 0.1
    The total reward returned is ***reward*** *=* *forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

    ## Starting State
    All position and velocity observations start in state `numpy.zeros((34,), dtype=float64)`
    with a noise added to the initial state for stochasticity.
    As seen before, the first 16 values in the state are positional and the last 18 values are velocity.
    A uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] is added to the positional values while a standard
    normal noise with a mean of 0 and standard deviation of `reset_noise_scale` is added to the
    initial velocity values of all zeros.
    While the the tendon gets reset to the size of 0.05 (m) and with 0 velocity.

    ## Episode End
    The episode truncates when the episode length is greater than 1000.

    ## Arguments
    No additional arguments are currently supported in pre-release.

    ## Version History
    * pre_release: part of `Gymnasium-Robotics/mamujoco`
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, render_mode: typing.Optional[str] = None):
        """Init.

        Args:
            render_mode: see [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
        """
        self._forward_reward_weight = 1
        self._ctrl_cost_weight = 0.1
        self._reset_noise_scale = 0.1

        observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float64
        )

        mujoco_env.MujocoEnv.__init__(
            self,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "assets",
                "coupled_half_cheetah.xml",
            ),
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
        )
        EzPickle.__init__(self, render_mode=render_mode)

    def step(self, action: np.ndarray):
        """Performs a single step given the `action`.

        Reward is the average reward of both half cheetahs (in the same structure as the single half Cheetah)
        Does never terminate (like Swimmer)
        """
        xposbefore1 = self.data.qpos[0]
        xposbefore2 = self.data.qpos[len(self.data.qpos) // 2]
        self.do_simulation(action, self.frame_skip)
        x_position_after1 = self.data.qpos[0]
        x_position_after2 = self.data.qpos[len(self.data.qpos) // 2]
        x_velocity1 = (x_position_after1 - xposbefore1) / self.dt  # velocity
        x_velocity2 = (x_position_after2 - xposbefore2) / self.dt  # velocity

        ctrl_cost1 = (
            self._ctrl_cost_weight * np.square(action[0 : len(action) // 2]).sum()
        )
        ctrl_cost2 = (
            self._ctrl_cost_weight * np.square(action[len(action) // 2 :]).sum()
        )

        forward_reward = self._forward_reward_weight * (x_velocity1 + x_velocity2) / 2.0

        observation = self._get_obs()
        reward = forward_reward - (ctrl_cost1 + ctrl_cost2) / 2.0
        terminal = False
        truncated = False
        info = {
            "x_position1": x_position_after1,
            "x_position2": x_position_after2,
            "x_velocity1": x_velocity1,
            "x_velocity2": x_velocity2,
            "reward_run": forward_reward,
            "reward_ctrl1": ctrl_cost1,
            "reward_ctrl2": ctrl_cost2,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminal, truncated, info

    def _get_obs(self) -> np.ndarray:
        # NOTE: does not return tendon data
        return np.concatenate(
            [
                self.data.qpos.flat[1:9],  # exclude rootx0
                self.data.qpos.flat[10:18],  # exclude rootx1
                self.data.qvel.flat,
                self.data.ten_J[0][:2],
                self.data.ten_J[0][9:11],
                self.data.ten_length,
                self.data.ten_velocity,
            ]
        )

    def reset_model(self) -> np.ndarray:
        """Resets the model in same way as the single half cheetah."""
        qpos = self.init_qpos + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        qvel = (
            self.init_qvel
            + self.np_random.standard_normal(self.model.nv) * self._reset_noise_scale
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        assert observation.shape == self.observation_space.shape
        return observation
