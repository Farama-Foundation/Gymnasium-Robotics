"""An Adroit arm environment with pen task using the Gymnasium API.

The code is inspired by the D4RL repository hosted on GitHub (https://github.com/Farama-Foundation/D4RL), published in the paper
'D4RL: Datasets for Deep Data-Driven Reinforcement Learning' by Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, Sergey Levine.

Original Author of the code: Justin Fu

The modifications made involve organizing the code into different files adding support for the Gymnasium API.

This project is covered by the Apache 2.0 License.
"""

from os import path
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.utils.rotations import euler2quat

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.0,
    "azimuth": -45.0,
}


class AdroitHandPenEnv(MujocoEnv, EzPickle):
    """
    ## Description

    This environment was introduced in ["Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations"](https://arxiv.org/abs/1709.10087)
    by Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine.

    The environment is based on the [Adroit manipulation platform](https://github.com/vikashplus/Adroit), a 28 degree of freedom system which consists of a 24 degrees of freedom
    ShadowHand and a 4 degree of freedom arm. The task to be completed consists on repositioning the blue pen to match the orientation of the green target. The base of the hand is fixed.
    The target is also randomized to cover all configurations. The task will be considered successful when the orientations match within tolerance

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (24,), float32)`. The control actions are absolute angular positions of the Adroit hand joints. The input of the control actions is set to a range between -1 and 1 by scaling the real actuator angle ranges in radians.
    The elements of the action array are the following:

    | Num | Action                                                                                  | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | --------------------------------------------------------------------------------------- | ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Angular position of the horizontal wrist joint (radial/ulnar deviation)                 | -1          | 1           | -0.524 (rad) | 0.175 (rad) | A_WRJ1                           | hinge | angle (rad) |
    | 1   | Angular position of the horizontal wrist joint (flexion/extension)                      | -1          | 1           | -0.79 (rad)  | 0.61 (rad)  | A_WRJ0                           | hinge | angle (rad) |
    | 2   | Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)    | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_FFJ3                           | hinge | angle (rad) |
    | 3   | Vertical angular position of the MCP joint of the forefinger (flexion/extension)        | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ2                           | hinge | angle (rad) |
    | 4   | Angular position of the PIP joint of the forefinger (flexion/extension)                 | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ1                           | hinge | angle (rad) |
    | 5   | Angular position of the DIP joint of the forefinger                                     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ0                           | hinge | angle (rad) |
    | 6   | Horizontal angular position of the MCP joint of the middle finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_MFJ3                           | hinge | angle (rad) |
    | 7   | Vertical angular position of the MCP joint of the middle finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ2                           | hinge | angle (rad) |
    | 8   | Angular position of the PIP joint of the middle finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ1                           | hinge | angle (rad) |
    | 9   | Angular position of the DIP joint of the middle finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ0                           | hinge | angle (rad) |
    | 10  | Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)   | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_RFJ3                           | hinge | angle (rad) |
    | 11  | Vertical angular position of the MCP joint of the ring finger (flexion/extension)       | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ2                           | hinge | angle (rad) |
    | 12  | Angular position of the PIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ1                           | hinge | angle (rad) |
    | 13  | Angular position of the DIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ0                           | hinge | angle (rad) |
    | 14  | Angular position of the CMC joint of the little finger                                  | -1          | 1           | 0 (rad)      | 0.7(rad)    | A_LFJ4                           | hinge | angle (rad) |
    | 15  | Horizontal angular position of the MCP joint of the little finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_LFJ3                           | hinge | angle (rad) |
    | 16  | Vertical angular position of the MCP joint of the little finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ2                           | hinge | angle (rad) |
    | 17  | Angular position of the PIP joint of the little finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ1                           | hinge | angle (rad) |
    | 18  | Angular position of the DIP joint of the little finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ0                           | hinge | angle (rad) |
    | 19  | Horizontal angular position of the CMC joint of the thumb finger                        | -1          | 1           | -1.047 (rad) | 1.047 (rad) | A_THJ4                           | hinge | angle (rad) |
    | 20  | Vertical Angular position of the CMC joint of the thumb finger                          | -1          | 1           | 0 (rad)      | 1.3 (rad)   | A_THJ3                           | hinge | angle (rad) |
    | 21  | Horizontal angular position of the MCP joint of the thumb finger (adduction/abduction)  | -1          | 1           | -0.26 (rad)  | 0.26(rad)   | A_THJ2                           | hinge | angle (rad) |
    | 22  | Vertical angular position of the MCP joint of the thumb finger (flexion/extension)      | -1          | 1           | -0.52 (rad)  | 0.52 (rad)  | A_THJ1                           | hinge | angle (rad) |
    | 23  | Angular position of the IP joint of the thumb finger (flexion/extension)                | -1          | 1           | -1.571 (rad) | 0 (rad)     | A_THJ0                           | hinge | angle (rad) |

    ## Observation Space

    The observation space is of the type `Box(-inf, inf, (45,), float64)`. It contains information about the angular position of the finger joints, the pose of the palm of the hand, as well as the pose of the real pen and target goal.

    | Num | Observation                                                                 | Min    | Max    | Joint Name (in corresponding XML file) | Site/Body Name (in corresponding XML file) | Joint Type| Unit                     |
    |-----|-----------------------------------------------------------------------------|--------|--------|----------------------------------------|--------------------------------------------|-----------|------------------------- |
    | 0   | Angular position of the horizontal wrist joint                              | -Inf   | Inf    | WRJ1                                   | -                                          | hinge     | angle (rad)              |
    | 1   | Angular position of the vertical wrist joint                                | -Inf   | Inf    | WRJ0                                   | -                                          | hinge     | angle (rad)              |
    | 2   | Horizontal angular position of the MCP joint of the forefinger              | -Inf   | Inf    | FFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 3   | Vertical angular position of the MCP joint of the forefinge                 | -Inf   | Inf    | FFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 4   | Angular position of the PIP joint of the forefinger                         | -Inf   | Inf    | FFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 5   | Angular position of the DIP joint of the forefinger                         | -Inf   | Inf    | FFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 6   | Horizontal angular position of the MCP joint of the middle finger           | -Inf   | Inf    | MFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 7   | Vertical angular position of the MCP joint of the middle finger             | -Inf   | Inf    | MFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 8   | Angular position of the PIP joint of the middle finger                      | -Inf   | Inf    | MFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 9   | Angular position of the DIP joint of the middle finger                      | -Inf   | Inf    | MFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 10  | Horizontal angular position of the MCP joint of the ring finger             | -Inf   | Inf    | RFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 11  | Vertical angular position of the MCP joint of the ring finger               | -Inf   | Inf    | RFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 12  | Angular position of the PIP joint of the ring finger                        | -Inf   | Inf    | RFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 13  | Angular position of the DIP joint of the ring finger                        | -Inf   | Inf    | RFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 14  | Angular position of the CMC joint of the little finger                      | -Inf   | Inf    | LFJ4                                   | -                                          | hinge     | angle (rad)              |
    | 15  | Horizontal angular position of the MCP joint of the little finger           | -Inf   | Inf    | LFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 16  | Vertical angular position of the MCP joint of the little finger             | -Inf   | Inf    | LFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 17  | Angular position of the PIP joint of the little finger                      | -Inf   | Inf    | LFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 18  | Angular position of the DIP joint of the little finger                      | -Inf   | Inf    | LFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 19  | Horizontal angular position of the CMC joint of the thumb finger            | -Inf   | Inf    | THJ4                                   | -                                          | hinge     | angle (rad)              |
    | 20  | Vertical Angular position of the CMC joint of the thumb finger              | -Inf   | Inf    | THJ3                                   | -                                          | hinge     | angle (rad)              |
    | 21  | Horizontal angular position of the MCP joint of the thumb finger            | -Inf   | Inf    | THJ2                                   | -                                          | hinge     | angle (rad)              |
    | 22  | Vertical angular position of the MCP joint of the thumb finger              | -Inf   | Inf    | THJ1                                   | -                                          | hinge     | angle (rad)              |
    | 23  | Angular position of the IP joint of the thumb finger                        | -Inf   | Inf    | THJ0                                   | -                                          | hinge     | angle (rad)              |
    | 24  | Position of the pen's center of mass in the x direction                     | -Inf   | Inf    | -                                      | Object                                     | -         | position (m)             |
    | 25  | Position of the pen's center of mass in the y direction                     | -Inf   | Inf    | -                                      | Object                                     | -         | position (m)             |
    | 26  | Position of the pen's center of mass in the z direction                     | -Inf   | Inf    | -                                      | Object                                     | -         | position (m)             |
    | 27  | Linear velocity of the pen in the x direction                               | -Inf   | Inf    | OBJTx                                  | -                                          | free      | velocity (m/s)           |
    | 28  | Linear velocity of the pen in the y direction                               | -Inf   | Inf    | OBJTy                                  | -                                          | free      | velocity (m/s)           |
    | 29  | Linear velocity of the pen in the z direction                               | -Inf   | Inf    | OBJTz                                  | -                                          | free      | velocity (m/s)           |
    | 30  | Angular velocity of the pen around x axis                                   | -Inf   | Inf    | OBJRx                                  | -                                          | free      | angular velocity (rad/s) |
    | 31  | Angular velocity of the pen around y axis                                   | -Inf   | Inf    | OBJRy                                  | -                                          | free      | angular velocity (rad/s) |
    | 32  | Angular velocity of the pen around z axis                                   | -Inf   | Inf    | OBJRz                                  | -                                          | free      | angular velocity (rad/s) |
    | 33  | Relative rotation of the pen's center of mass with respect to the x axis    | -Inf   | Inf    | -                                      | object_top,object_bottom                   | -         | angle (rad)              |
    | 34  | Relative rotation of the pen's center of mass with respect to the y axis    | -Inf   | Inf    | -                                      | object_top,object_bottom                   | -         | angle (rad)              |
    | 35  | Relative rotation of the pen's center of mass with respect to the z axis    | -Inf   | Inf    | -                                      | object_top,object_bottom                   | -         | angle (rad)              |
    | 36  | Relative rotation of the target's center of mass with respect to the x axis | -Inf   | Inf    | -                                      | target_top,target_bottom                   | -         | angle (rad)              |
    | 37  | Relative rotation of the target's center of mass with respect to the y axis | -Inf   | Inf    | -                                      | target_top,target_bottom                   | -         | angle (rad)              |
    | 38  | Relative rotation of the target's center of mass with respect to the z axis | -Inf   | Inf    | -                                      | target_top,target_bottom                   | -         | angle (rad)              |
    | 39  | x linear distance from pen to target goal                                   | -Inf   | Inf    | -                                      | -                                          | -         | position (m)             |
    | 40  | y linear distance from pen to target goal                                   | -Inf   | Inf    | -                                      | -                                          | -         | position (m)             |
    | 41  | z linear distance from pen to target goal                                   | -Inf   | Inf    | -                                      | -                                          | -         | position (m)             |
    | 42  | Rotational distance from pen to target goal with respect to the x axis      | -Inf   | Inf    | -                                      | -                                          | -         | angle (rad)              |
    | 43  | Rotational distance from pen to target goal with respect to the x axis      | -Inf   | Inf    | -                                      | -                                          | -         | angle (rad)              |
    | 44  | Rotational distance from pen to target goal with respect to the x axis      | -Inf   | Inf    | -                                      | -                                          | -         | angle (rad)              |

    ## Rewards

    The environment can be initialized in either a `dense` or `sparse` reward variant.

    In the `dense` reward setting, the environment returns a `dense` reward function that consists of the following parts:
    - `target_distance`: increasing negative reward the further away the pen is from its target. This is computed as the 3 dimensional Euclidean distance between both body frames.
        This penalty is scaled by a factor of `0.1` in the final reward.
    - `orientation_similarity`: add the dot product between the target's and real pen orientation.
    - `close_to_target`: bonus reward for the pen being close to the target orientation. If the dot product between both ortientations is greater than `0.9` and the Euclidean
        distance less than `0.075` add a `10` reward, if the same distance holds and the orientation dot product is greater than `0.95` add `50`.
    - `dropping_pen`: If the pen drops from the hand (pen's height less than `0.075`) add a negative reward of `5`.

    The `sparse` reward variant of the environment can be initialized by calling `gym.make('AdroitHandPenSparse-v1')`.
    In this variant, the environment returns a reward of 10 for environment success and -0.1 otherwise.

    ## Starting State

    The real pen is reset to the palm of the Adroit arm. The target orientation of the pen is then randomly selected from a uniform distribution with range `[-1,1]` radians.
    Only roll and pitch are randomly selected. The initial position of the target is `(x,y,z)=(0,-0.2,0.25)`.

    The joint values of the environment are deterministically initialized to a zero.

    For reproducibility, the starting state of the environment can also be set when calling `env.reset()` by passing the `options` dictionary argument (https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)
    with the `initial_state_dict` key. The `initial_state_dict` key must be a dictionary with the following items:

    * `qpos`: np.ndarray with shape `(30,)`, MuJoCo simulation joint positions
    * `qvel`: np.ndarray with shape `(30,)`, MuJoCo simulation joint velocities
    * `desired_orien`: np.ndarray with shape `(4,)`, quaternion values of the target pen orientation

    The state of the simulation can also be set at any step with the `env.set_env_state(initial_state_dict)` method.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 200 timesteps.
    The episode will be `terminated` when the Euclidean distancd to the target is less than `0.075`, and the dot product of the pen's and target orientatin
    is greater than `0.95`.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 400 make the environment as follows:

    ```python
    import gymnasium as gym

    env = gym.make('AdroitHandPen-v1', max_episode_steps=400)
    ```

    ## Version History

    * v1: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type: str = "dense", **kwargs):
        self.pen_length = 1.0
        self.tar_length = 1.0

        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            "../assets/adroit_hand/adroit_pen.xml",
        )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.model)

        # whether to have sparse rewards
        if reward_type.lower() == "dense":
            self.sparse_reward = False
        elif reward_type.lower() == "sparse":
            self.sparse_reward = True
        else:
            raise ValueError(
                f"Unknown reward type, expected `dense` or `sparse` but got {reward_type}"
            )

        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )

        # change actuator sensitivity
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"]
            + 1,
            :3,
        ] = np.array([10, 0, 0])
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"]
            + 1,
            :3,
        ] = np.array([1, 0, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"]
            + 1,
            :3,
        ] = np.array([0, -10, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"]
            + 1,
            :3,
        ] = np.array([0, -1, 0])

        self.target_obj_body_id = self._model_names.body_name2id["target"]
        self.S_grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.obj_body_id = self._model_names.body_name2id["Object"]
        self.eps_ball_site_id = self._model_names.site_name2id["eps_ball"]
        self.obj_t_site_id = self._model_names.site_name2id["object_top"]
        self.obj_b_site_id = self._model_names.site_name2id["object_bottom"]
        self.tar_t_site_id = self._model_names.site_name2id["target_top"]
        self.tar_b_site_id = self._model_names.site_name2id["target_bottom"]

        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )

        self._state_space = spaces.Dict(
            {
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64
                ),
                "desired_orien": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                ),
            }
        )

        EzPickle.__init__(self, **kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        a = self.act_mean + a * self.act_rng  # mean center and scale
        self.do_simulation(a, self.frame_skip)

        obs = self._get_obs()

        obj_pos = self.data.xpos[self.obj_body_id].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_site_id].ravel()
        obj_orien = (
            self.data.site_xpos[self.obj_t_site_id]
            - self.data.site_xpos[self.obj_b_site_id]
        ) / self.pen_length
        desired_orien = (
            self.data.site_xpos[self.tar_t_site_id]
            - self.data.site_xpos[self.tar_b_site_id]
        ) / self.tar_length

        # compute the sparse reward variant first
        goal_distance = np.linalg.norm(obj_pos - desired_loc)
        orien_similarity = np.dot(obj_orien, desired_orien)
        goal_achieved = goal_distance < 0.075 and orien_similarity > 0.95
        reward = 10.0 if goal_achieved else -0.1

        # goal_failed = obj_pos[2] < 0.075

        # override reward if not sparse reward
        if not self.sparse_reward:
            reward = -goal_distance + orien_similarity

            # bonus for being close to desired orientation
            if goal_distance < 0.075 and orien_similarity > 0.9:
                reward += 10
            if goal_distance < 0.075 and orien_similarity > 0.95:
                reward += 50

            # penalty for dropping the pen
            if obj_pos[2] < 0.075:
                reward -= 5

        if self.render_mode == "human":
            self.render()

        return (
            obs,
            reward,
            # goal_failed or goal_achieved,
            False,
            False,
            dict(success=goal_achieved),
        )

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.xpos[self.obj_body_id].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_site_id].ravel()
        obj_orien = (
            self.data.site_xpos[self.obj_t_site_id]
            - self.data.site_xpos[self.obj_b_site_id]
        ) / self.pen_length
        desired_orien = (
            self.data.site_xpos[self.tar_t_site_id]
            - self.data.site_xpos[self.tar_b_site_id]
        ) / self.tar_length

        return np.concatenate(
            [
                qpos[:-6],
                obj_pos,
                obj_vel,
                obj_orien,
                desired_orien,
                obj_pos - desired_pos,
                obj_orien - desired_orien,
            ]
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        obs, info = super().reset(seed=seed)
        if options is not None and "initial_state_dict" in options:
            self.set_env_state(options["initial_state_dict"])
            obs = self._get_obs()

        return obs, info

    def reset_model(self):
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_body_id] = euler2quat(desired_orien)

        self.set_state(self.init_qpos, self.init_qvel)

        self.pen_length = np.linalg.norm(
            self.data.site_xpos[self.obj_t_site_id]
            - self.data.site_xpos[self.obj_b_site_id]
        )
        self.tar_length = np.linalg.norm(
            self.data.site_xpos[self.tar_t_site_id]
            - self.data.site_xpos[self.tar_b_site_id]
        )

        obs = self._get_obs()
        return obs

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_body_id].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        assert self._state_space.contains(
            state_dict
        ), f"The state dictionary {state_dict} must be a member of {self._state_space}."
        qp = state_dict["qpos"]
        qv = state_dict["qvel"]

        self.model.body_quat[self.target_obj_body_id] = state_dict["desired_orien"]
        self.set_state(qp, qv)
