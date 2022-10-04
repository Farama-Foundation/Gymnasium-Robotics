import os
from typing import Union

import numpy as np
from gymnasium import error
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.hand_env import MujocoHandEnv, MujocoPyHandEnv
from gymnasium_robotics.utils import rotations


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])
    quat /= np.linalg.norm(quat)
    return quat


# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join("hand", "manipulate_block.xml")
MANIPULATE_EGG_XML = os.path.join("hand", "manipulate_egg.xml")
MANIPULATE_PEN_XML = os.path.join("hand", "manipulate_pen.xml")


def get_base_manipulate_env(HandEnvClass: Union[MujocoHandEnv, MujocoPyHandEnv]):
    """Factory function that returns a BaseManipulateEnv class that inherits
    from MujocoPyHandEnv or MujocoHandEnv depending on the mujoco python bindings.
    """

    class BaseManipulateEnv(HandEnvClass):
        def __init__(
            self,
            target_position,
            target_rotation,
            target_position_range,
            reward_type,
            initial_qpos=None,
            randomize_initial_position=True,
            randomize_initial_rotation=True,
            distance_threshold=0.01,
            rotation_threshold=0.1,
            n_substeps=20,
            relative_control=False,
            ignore_z_target_rotation=False,
            **kwargs,
        ):
            """Initializes a new Hand manipulation environment.

            Args:
                model_path (string): path to the environments XML file
                target_position (string): the type of target position:
                    - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                    - fixed: target position is set to the initial position of the object
                    - random: target position is fully randomized according to target_position_range
                target_rotation (string): the type of target rotation:
                    - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                    - fixed: target rotation is set to the initial rotation of the object
                    - xyz: fully randomized target rotation around the X, Y and Z axis
                    - z: fully randomized target rotation around the Z axis
                    - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
                ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
                target_position_range (np.array of shape (3, 2)): range of the target_position randomization
                reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
                randomize_initial_position (boolean): whether or not to randomize the initial position of the object
                randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
                distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
                rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
                n_substeps (int): number of substeps the simulation runs on every call to step
                relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
            """
            self.target_position = target_position
            self.target_rotation = target_rotation
            self.target_position_range = target_position_range
            self.parallel_quats = [
                rotations.euler2quat(r) for r in rotations.get_parallel_rotations()
            ]
            self.randomize_initial_rotation = randomize_initial_rotation
            self.randomize_initial_position = randomize_initial_position
            self.distance_threshold = distance_threshold
            self.rotation_threshold = rotation_threshold
            self.reward_type = reward_type
            self.ignore_z_target_rotation = ignore_z_target_rotation

            assert self.target_position in ["ignore", "fixed", "random"]
            assert self.target_rotation in ["ignore", "fixed", "xyz", "z", "parallel"]
            initial_qpos = initial_qpos or {}

            super().__init__(
                n_substeps=n_substeps,
                initial_qpos=initial_qpos,
                relative_control=relative_control,
                **kwargs,
            )

        def _goal_distance(self, goal_a, goal_b):
            assert goal_a.shape == goal_b.shape
            assert goal_a.shape[-1] == 7

            d_pos = np.zeros_like(goal_a[..., 0])
            d_rot = np.zeros_like(goal_b[..., 0])
            if self.target_position != "ignore":
                delta_pos = goal_a[..., :3] - goal_b[..., :3]
                d_pos = np.linalg.norm(delta_pos, axis=-1)

            if self.target_rotation != "ignore":
                quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

                if self.ignore_z_target_rotation:
                    # Special case: We want to ignore the Z component of the rotation.
                    # This code here assumes Euler angles with xyz convention. We first transform
                    # to euler, then set the Z component to be equal between the two, and finally
                    # transform back into quaternions.
                    euler_a = rotations.quat2euler(quat_a)
                    euler_b = rotations.quat2euler(quat_b)
                    euler_a[2] = euler_b[2]
                    quat_a = rotations.euler2quat(euler_a)

                # Subtract quaternions and extract angle between them.
                quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
                angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))
                d_rot = angle_diff
            assert d_pos.shape == d_rot.shape
            return d_pos, d_rot

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            if self.reward_type == "sparse":
                success = self._is_success(achieved_goal, goal).astype(np.float32)
                return success - 1.0
            else:
                d_pos, d_rot = self._goal_distance(achieved_goal, goal)
                # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
                # dominated by `d_rot` (in radians).
                return -(10.0 * d_pos + d_rot)

        # RobotEnv methods
        # ----------------------------

        def _is_success(self, achieved_goal, desired_goal):
            d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
            achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
            achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
            achieved_both = achieved_pos * achieved_rot
            return achieved_both

    return BaseManipulateEnv


class MujocoManipulateEnv(get_base_manipulate_env(MujocoHandEnv)):
    def _get_achieved_goal(self):
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object:joint")
        assert object_qpos.shape == (7,)
        return object_qpos

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self._mujoco.mj_forward(self.model, self.data)
        initial_qpos = self._utils.get_joint_qpos(
            self.model, self.data, "object:joint"
        ).copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.target_rotation == "z":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "parallel":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[
                    self.np_random.integers(len(self.parallel_quats))
                ]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ["xyz", "ignore"]:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1.0, 1.0, size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "fixed":
                pass
            else:
                raise error.Error(
                    f'Unknown target_rotation option "{self.target_rotation}".'
                )

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != "fixed":
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])

        self._utils.set_joint_qpos(self.model, self.data, "object:joint", initial_qpos)

        def is_on_palm():
            self._mujoco.mj_forward(self.model, self.data)
            cube_middle_idx = self._model_names._site_name2id["object:center"]
            cube_middle_pos = self.data.site_xpos[cube_middle_idx]
            is_on_palm = cube_middle_pos[2] > 0.04
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(20))
            try:
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
            except Exception:
                return False

        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == "random":
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(
                self.target_position_range[:, 0], self.target_position_range[:, 1]
            )
            assert offset.shape == (3,)
            target_pos = (
                self._utils.get_joint_qpos(self.model, self.data, "object:joint")[:3]
                + offset
            )
        elif self.target_position in ["ignore", "fixed"]:
            target_pos = self._utils.get_joint_qpos(
                self.model, self.data, "object:joint"
            )[:3]
        else:
            raise error.Error(
                f'Unknown target_position option "{self.target_position}".'
            )
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == "z":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == "parallel":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[
                self.np_random.integers(len(self.parallel_quats))
            ]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == "xyz":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1.0, 1.0, size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ["ignore", "fixed"]:
            target_quat = self.data.get_joint_qpos("object:joint")
        else:
            raise error.Error(
                f'Unknown target_rotation option "{self.target_rotation}".'
            )
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == "ignore":
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15

        self._utils.set_joint_qpos(self.model, self.data, "target:joint", goal)
        self._utils.set_joint_qvel(self.model, self.data, "target:joint", np.zeros(6))

        if "object_hidden" in self._model_names.geom_names:
            hidden_id = self._model_names.geom_name2id["object_hidden"]
            self.model.geom_rgba[hidden_id, 3] = 1.0
        self._mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        object_qvel = self._utils.get_joint_qvel(self.model, self.data, "object:joint")
        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation

        observation = np.concatenate(
            [robot_qpos, robot_qvel, object_qvel, achieved_goal]
        )
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }


class MujocoPyManipulateEnv(get_base_manipulate_env(MujocoPyHandEnv)):
    def _get_achieved_goal(self):
        # Object position and rotation.
        object_qpos = self.sim.data.get_joint_qpos("object:joint")

        assert object_qpos.shape == (7,)
        return object_qpos

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos("object:joint").copy()

        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.target_rotation == "z":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "parallel":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[
                    self.np_random.integers(len(self.parallel_quats))
                ]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ["xyz", "ignore"]:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1.0, 1.0, size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "fixed":
                pass
            else:
                raise error.Error(
                    f'Unknown target_rotation option "{self.target_rotation}".'
                )

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != "fixed":
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])

        self.sim.data.set_joint_qpos("object:joint", initial_qpos)

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id("object:center")
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]

            is_on_palm = cube_middle_pos[2] > 0.04
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(20))
            try:
                self.sim.step()
            except self._mujoco_py.MujocoException:
                return False

        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == "random":
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(
                self.target_position_range[:, 0], self.target_position_range[:, 1]
            )
            assert offset.shape == (3,)
            target_pos = self.sim.data.get_joint_qpos("object:joint")[:3] + offset

        elif self.target_position in ["ignore", "fixed"]:
            target_pos = self.sim.data.get_joint_qpos("object:joint")[:3]
        else:
            raise error.Error(
                f'Unknown target_position option "{self.target_position}".'
            )
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == "z":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == "parallel":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0.0, 0.0, 1.0])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[
                self.np_random.integers(len(self.parallel_quats))
            ]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == "xyz":
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1.0, 1.0, size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ["ignore", "fixed"]:
            target_quat = self.sim.data.get_joint_qpos("object:joint")
        else:
            raise error.Error(
                f'Unknown target_rotation option "{self.target_rotation}".'
            )
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == "ignore":
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
        self.sim.data.set_joint_qpos("target:joint", goal)
        self.sim.data.set_joint_qvel("target:joint", np.zeros(6))

        if "object_hidden" in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id("object_hidden")
            self.sim.model.geom_rgba[hidden_id, 3] = 1.0
        self.sim.forward()

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel("object:joint")

        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        observation = np.concatenate(
            [robot_qpos, robot_qvel, object_qvel, achieved_goal]
        )
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }


class MujocoHandBlockEnv(MujocoManipulateEnv):
    """
    ### Description

    This environment was introduced in ["Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"](https://arxiv.org/abs/1802.09464).

    The environment is based on the same robot hand as in the `HandReach` environment, the [Shadow Dexterous Hand](https://www.shadowrobot.com/). In this task a block is placed on the palm of the hand. The task is to then manipulate the
    block such that a target pose is achieved. The goal is 7-dimensional and includes the target position (in Cartesian coordinates) and target rotation (in quaternions). In addition, variations of this environment can be used with increasing
    levels of difficulty:

    * `HandManipulateBlockRotateZ-v1`: Random target rotation around the *z* axis of the block. No target position.
    * `HandManipulateBlockRotateParallel-v1`: Random target rotation around the *z* axis of the block and axis-aligned target rotations for the *x* and *y* axes. No target position.
    * `HandManipulateBlockRotateXYZ-v1`: Random target rotation for all axes of the block. No target position.
    * `HandManipulateBlockFull-v1`: Random target rotation for all axes of the block. Random target position.

    ### Action Space

    The action space is a `Box(-1.0, 1.0, (20,), float32)`. The control actions are absolute angular positions of the actuated joints (non-coupled). The input of the control actions is set to a range between -1 and 1 by scaling the actual actuator angle ranges.
    The elements of the action array are the following:

    | Num | Action                                                                                  | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | --------------------------------------------------------------------------------------- | ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Angular position of the horizontal wrist joint (radial/ulnar deviation)                 | -1          | 1           | -0.489 (rad) | 0.14 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
    | 1   | Angular position of the horizontal wrist joint (flexion/extension)                      | -1          | 1           | -0.698 (rad) | 0.489 (rad) | robot0:A_WRJ0                    | hinge | angle (rad) |
    | 2   | Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)    | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_FFJ3                    | hinge | angle (rad) |
    | 3   | Vertical angular position of the MCP joint of the forefinger (flexion/extension)        | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_FFJ2                    | hinge | angle (rad) |
    | 4   | Angular position of the PIP joint of the forefinger (flexion/extension)                 | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_FFJ1                    | hinge | angle (rad) |
    | 5   | Horizontal angular position of the MCP joint of the middle finger (adduction/abduction) | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_MFJ3                    | hinge | angle (rad) |
    | 6   | Vertical angular position of the MCP joint of the middle finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_MFJ2                    | hinge | angle (rad) |
    | 7   | Angular position of the PIP joint of the middle finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_MFJ1                    | hinge | angle (rad) |
    | 8   | Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)   | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_RFJ3                    | hinge | angle (rad) |
    | 9   | Vertical angular position of the MCP joint of the ring finger (flexion/extension)       | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_RFJ2                    | hinge | angle (rad) |
    | 10  | Angular position of the PIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_RFJ1                    | hinge | angle (rad) |
    | 11  | Angular position of the CMC joint of the little finger                                  | -1          | 1           | 0 (rad)      | 0.785(rad)  | robot0:A_LFJ4                    | hinge | angle (rad) |
    | 12  | Horizontal angular position of the MCP joint of the little finger (adduction/abduction) | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_LFJ3                    | hinge | angle (rad) |
    | 13  | Vertical angular position of the MCP joint of the little finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_LFJ2                    | hinge | angle (rad) |
    | 14  | Angular position of the PIP joint of the little finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_LFJ1                    | hinge | angle (rad) |
    | 15  | Horizontal angular position of the CMC joint of the thumb finger                        | -1          | 1           | -1.047 (rad) | 1.047 (rad) | robot0:A_THJ4                    | hinge | angle (rad) |
    | 16  | Vertical Angular position of the CMC joint of the thumb finger                          | -1          | 1           | 0 (rad)      | 1.222 (rad) | robot0:A_THJ3                    | hinge | angle (rad) |
    | 17  | Horizontal angular position of the MCP joint of the thumb finger (adduction/abduction)  | -1          | 1           | -0.209 (rad) | 0.209(rad)  | robot0:A_THJ2                    | hinge | angle (rad) |
    | 18  | Vertical angular position of the MCP joint of the thumb finger (flexion/extension)      | -1          | 1           | -0.524 (rad) | 0.524 (rad) | robot0:A_THJ1                    | hinge | angle (rad) |
    | 19  | Angular position of the IP joint of the thumb finger (flexion/extension)                | -1          | 1           | -1.571 (rad) | 0 (rad)     | robot0:A_THJ0                    | hinge | angle (rad) |


    ### Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and block states, as well as information about the goal. The dictionary consists of the following 3 keys:

    - `observation`: its value is an `ndarray` of shape `(61,)`. It consists of kinematic information of the block object and finger joints. The elements of the array correspond to the following:

        | Num | Observation                                                       | Min    | Max    | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
        |-----|-------------------------------------------------------------------|--------|--------|----------------------------------------|----------|------------------------- |
        | 0   | Angular position of the horizontal wrist joint                    | -Inf   | Inf    | robot0:WRJ1                            | hinge    | angle (rad)              |
        | 1   | Angular position of the vertical wrist joint                      | -Inf   | Inf    | robot0:WRJ0                            | hinge    | angle (rad)              |
        | 2   | Horizontal angular position of the MCP joint of the forefinger    | -Inf   | Inf    | robot0:FFJ3                            | hinge    | angle (rad)              |
        | 3   | Vertical angular position of the MCP joint of the forefinge       | -Inf   | Inf    | robot0:FFJ2                            | hinge    | angle (rad)              |
        | 4   | Angular position of the PIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ1                            | hinge    | angle (rad)              |
        | 5   | Angular position of the DIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ0                            | hinge    | angle (rad)              |
        | 6   | Horizontal angular position of the MCP joint of the middle finger | -Inf   | Inf    | robot0:MFJ3                            | hinge    | angle (rad)              |
        | 7   | Vertical angular position of the MCP joint of the middle finger   | -Inf   | Inf    | robot0:MFJ2                            | hinge    | angle (rad)              |
        | 8   | Angular position of the PIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ1                            | hinge    | angle (rad)              |
        | 9   | Angular position of the DIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ0                            | hinge    | angle (rad)              |
        | 10  | Horizontal angular position of the MCP joint of the ring finger   | -Inf   | Inf    | robot0:RFJ3                            | hinge    | angle (rad)              |
        | 11  | Vertical angular position of the MCP joint of the ring finger     | -Inf   | Inf    | robot0:RFJ2                            | hinge    | angle (rad)              |
        | 12  | Angular position of the PIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ1                            | hinge    | angle (rad)              |
        | 13  | Angular position of the DIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ0                            | hinge    | angle (rad)              |
        | 14  | Angular position of the CMC joint of the little finger            | -Inf   | Inf    | robot0:LFJ4                            | hinge    | angle (rad)              |
        | 15  | Horizontal angular position of the MCP joint of the little finger | -Inf   | Inf    | robot0:LFJ3                            | hinge    | angle (rad)              |
        | 16  | Vertical angular position of the MCP joint of the little finger   | -Inf   | Inf    | robot0:LFJ2                            | hinge    | angle (rad)              |
        | 17  | Angular position of the PIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ1                            | hinge    | angle (rad)              |
        | 18  | Angular position of the DIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ0                            | hinge    | angle (rad)              |
        | 19  | Horizontal angular position of the CMC joint of the thumb finger  | -Inf   | Inf    | robot0:THJ4                            | hinge    | angle (rad)              |
        | 20  | Vertical Angular position of the CMC joint of the thumb finger    | -Inf   | Inf    | robot0:THJ3                            | hinge    | angle (rad)              |
        | 21  | Horizontal angular position of the MCP joint of the thumb finger  | -Inf   | Inf    | robot0:THJ2                            | hinge    | angle (rad)              |
        | 22  | Vertical angular position of the MCP joint of the thumb finger    | -Inf   | Inf    | robot0:THJ1                            | hinge    | angle (rad)              |
        | 23  | Angular position of the IP joint of the thumb finger              | -Inf   | Inf    | robot0:THJ0                            | hinge    | angle (rad)              |
        | 24  | Angular velocity of the horizontal wrist joint                    | -Inf   | Inf    | robot0:WRJ1                            | hinge    | angular velocity (rad/s) |
        | 25  | Angular velocity of the vertical wrist joint                      | -Inf   | Inf    | robot0:WRJ0                            | hinge    | angular velocity (rad/s) |
        | 26  | Horizontal angular velocity of the MCP joint of the forefinger    | -Inf   | Inf    | robot0:FFJ3                            | hinge    | angular velocity (rad/s) |
        | 27  | Vertical angular velocity of the MCP joint of the forefinge       | -Inf   | Inf    | robot0:FFJ2                            | hinge    | angular velocity (rad/s) |
        | 28  | Angular velocity of the PIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ1                            | hinge    | angular velocity (rad/s) |
        | 29  | Angular velocity of the DIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ0                            | hinge    | angular velocity (rad/s) |
        | 30  | Horizontal angular velocity of the MCP joint of the middle finger | -Inf   | Inf    | robot0:MFJ3                            | hinge    | angular velocity (rad/s) |
        | 31  | Vertical angular velocity of the MCP joint of the middle finger   | -Inf   | Inf    | robot0:MFJ2                            | hinge    | angular velocity (rad/s) |
        | 32  | Angular velocity of the PIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ1                            | hinge    | angular velocity (rad/s) |
        | 33  | Angular velocity of the DIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ0                            | hinge    | angular velocity (rad/s) |
        | 34  | Horizontal angular velocity of the MCP joint of the ring finger   | -Inf   | Inf    | robot0:RFJ3                            | hinge    | angular velocity (rad/s) |
        | 35  | Vertical angular velocity of the MCP joint of the ring finger     | -Inf   | Inf    | robot0:RFJ2                            | hinge    | angular velocity (rad/s) |
        | 36  | Angular velocity of the PIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ1                            | hinge    | angular velocity (rad/s) |
        | 37  | Angular velocity of the DIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ0                            | hinge    | angular velocity (rad/s) |
        | 38  | Angular velocity of the CMC joint of the little finger            | -Inf   | Inf    | robot0:LFJ4                            | hinge    | angular velocity (rad/s) |
        | 39  | Horizontal angular velocity of the MCP joint of the little finger | -Inf   | Inf    | robot0:LFJ3                            | hinge    | angular velocity (rad/s) |
        | 40  | Vertical angular velocity of the MCP joint of the little finger   | -Inf   | Inf    | robot0:LFJ2                            | hinge    | angular velocity (rad/s) |
        | 41  | Angular velocity of the PIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ1                            | hinge    | angular velocity (rad/s) |
        | 42  | Angular velocity of the DIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ0                            | hinge    | angular velocity (rad/s) |
        | 43  | Horizontal angular velocity of the CMC joint of the thumb finger  | -Inf   | Inf    | robot0:THJ4                            | hinge    | angular velocity (rad/s) |
        | 44  | Vertical Angular velocity of the CMC joint of the thumb finger    | -Inf   | Inf    | robot0:THJ3                            | hinge    | angular velocity (rad/s) |
        | 45  | Horizontal angular velocity of the MCP joint of the thumb finger  | -Inf   | Inf    | robot0:THJ2                            | hinge    | angular velocity (rad/s) |
        | 46  | Vertical angular position of the MCP joint of the thumb finger    | -Inf   | Inf    | robot0:THJ1                            | hinge    | angular velocity (rad/s) |
        | 47  | Angular velocity of the IP joint of the thumb finger              | -Inf   | Inf    | robot0:THJ0                            | hinge    | angular velocity (rad/s) |
        | 48  | Linear velocity of the block in x direction                       | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 49  | Linear velocity of the block in y direction                       | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 50  | Linear velocity of the block in z direction                       | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 51  | Angular velocity of the block in x axis                           | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 52  | Angular velocity of the block in y axis                           | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 53  | Angular velocity of the block in z axis                           | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 54  | Position of the block in the x coordinate                         | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 55  | Position of the block in the y coordinate                         | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 56  | Position of the block in the z coordinate                         | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 57  | w component of the quaternion orientation of the block            | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 58  | x component of the quaternion orientation of the block            | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 59  | y component of the quaternion orientation of the block            | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 60  | z component of the quaternion orientation of the block            | -Inf   | Inf    | object:joint                           | free     | -                        |

    - `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 7-dimensional `ndarray`, `(7,)`, that consists of the pose information of the block. The elements of the array are the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
        | 0   | Target x coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 1   | Target y coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 2   | Target z coordinate of the block                                                                                                      | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 3   | Target w component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
        | 4   | Target x component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
        | 5   | Target y component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |
        | 6   | Target z component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | target:joint                           | free       | -            |


    - `achieved_goal`: this key represents the current state of the block, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER).
    The value is an `ndarray` with shape `(7,)`. The elements of the array are the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
        | 0   | Current x coordinate of the block                                                                                                      | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 1   | Current y coordinate of the block                                                                                                      | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 2   | Current z coordinate of the block                                                                                                      | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 3   | Current w component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | object:joint                           | free       | -            |
        | 4   | Current x component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | object:joint                           | free       | -            |
        | 5   | Current y component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | object:joint                           | free       | -            |
        | 6   | Current z component of the quaternion orientation of the block                                                                         | -Inf   | Inf    | object:joint                           | free       | -            |


    ### Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the block hasn't reached its final target pose, and `0` if the block is in its final target pose. The block is considered to have reached its final goal if the theta angle difference (theta angle of the
    [3D axis angle representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) is less than 0.1 and if the Euclidean distance to the target position is also less than 0.01 m.
    - *dense*: the returned reward is the negative summation of the Euclidean distance to the block's target and the theta angle difference to the target orientation. The positional distance is multiplied by a factor of 10 to avoid being dominated by the rotational difference.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `HandManipulateBlock-v1`. However, for `dense`
    reward the id must be modified to `HandManipulateBlockDense-v1` and initialized as follows:

    ```python
    import gymnasium as gym

    env = gym.make('HandManipulateBlock-v1')
    ```

    The rest of the id's of the other environment variations follow the same convention to select between a sparse or dense reward function.

    ### Starting State

    When the environment is reset the joints of the hand are initialized to their resting position with a 0 displacement. The blocks position and orientation are randomly selected. The initial position is set to `(x,y,z)=(1, 0.87, 0.2)` and an offset is added to each coordinate
    sampled from a normal distribution with 0 mean and 0.005 standard deviation.
    While the initial orientation is set to `(w,x,y,z)=(1,0,0,0)` and an axis is randomly selected depending on the environment variation to add an angle offset sampled from a uniform distribution with range `[-pi, pi]`.

    The target pose of the block is obtained by adding a random offset to the initial block pose. For the position the offset is sampled from a uniform distribution with range `[(x_min, x_max), (y_min,y_max), (z_min, z_max)] = [(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]`. The orientation
    offset is sampled from a uniform distribution with range `[-pi,pi]` and added to one of the Euler axis depending on the environment variation.


    ### Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ### Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

    ```python
    import gymnasium as gym

    env = gym.make('HandManipulateBlock-v1', max_episode_steps=100)
    ```

    The same applies for the other environment variations.

    ### Version History

    * v1: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: the environment depends on `mujoco_py` which is no longer maintained.

    """

    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_BLOCK_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)


class MujocoPyHandBlockEnv(MujocoPyManipulateEnv, EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoPyManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_BLOCK_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)


class MujocoHandEggEnv(MujocoManipulateEnv, EzPickle):
    """
    ### Description

    This environment was introduced in ["Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"](https://arxiv.org/abs/1802.09464).

    The environment is based on the same robot hand as in the `HandReach` environment, the [Shadow Dexterous Hand](https://www.shadowrobot.com/). The task to be solved is
    very similar to that in the `HandManipulateBlock` environment, but in this case an egg-shaped object is placed on the palm of the hand. The task is to then manipulate
    the object such that a target pose is achieved. The goal is 7-dimensional and includes the target position (in Cartesian coordinates) and target rotation (in quaternions).
    In addition, variations of this environment can be used with increasing levels of difficulty:

    * `HandManipulateEggRotate-v1`: Random target rotation for all axes of the egg. No target position.
    * `HandManipulateEggFull-v1`:  Random target rotation for all axes of the egg. Random target position.

    ### Action Space

    The action space is a `Box(-1.0, 1.0, (20,), float32)`. The control actions are absolute angular positions of the actuated joints (non-coupled). The input of the control
    actions is set to a range between -1 and 1 by scaling the actual actuator angle ranges. The elements of the action array are the following:

    | Num | Action                                                                                  | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | --------------------------------------------------------------------------------------- | ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Angular position of the horizontal wrist joint (radial/ulnar deviation)                 | -1          | 1           | -0.489 (rad) | 0.14 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
    | 1   | Angular position of the horizontal wrist joint (flexion/extension)                      | -1          | 1           | -0.698 (rad) | 0.489 (rad) | robot0:A_WRJ0                    | hinge | angle (rad) |
    | 2   | Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)    | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_FFJ3                    | hinge | angle (rad) |
    | 3   | Vertical angular position of the MCP joint of the forefinger (flexion/extension)        | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_FFJ2                    | hinge | angle (rad) |
    | 4   | Angular position of the PIP joint of the forefinger (flexion/extension)                 | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_FFJ1                    | hinge | angle (rad) |
    | 5   | Horizontal angular position of the MCP joint of the middle finger (adduction/abduction) | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_MFJ3                    | hinge | angle (rad) |
    | 6   | Vertical angular position of the MCP joint of the middle finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_MFJ2                    | hinge | angle (rad) |
    | 7   | Angular position of the PIP joint of the middle finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_MFJ1                    | hinge | angle (rad) |
    | 8   | Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)   | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_RFJ3                    | hinge | angle (rad) |
    | 9   | Vertical angular position of the MCP joint of the ring finger (flexion/extension)       | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_RFJ2                    | hinge | angle (rad) |
    | 10  | Angular position of the PIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_RFJ1                    | hinge | angle (rad) |
    | 11  | Angular position of the CMC joint of the little finger                                  | -1          | 1           | 0 (rad)      | 0.785(rad)  | robot0:A_LFJ4                    | hinge | angle (rad) |
    | 12  | Horizontal angular position of the MCP joint of the little finger (adduction/abduction) | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_LFJ3                    | hinge | angle (rad) |
    | 13  | Vertical angular position of the MCP joint of the little finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_LFJ2                    | hinge | angle (rad) |
    | 14  | Angular position of the PIP joint of the little finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_LFJ1                    | hinge | angle (rad) |
    | 15  | Horizontal angular position of the CMC joint of the thumb finger                        | -1          | 1           | -1.047 (rad) | 1.047 (rad) | robot0:A_THJ4                    | hinge | angle (rad) |
    | 16  | Vertical Angular position of the CMC joint of the thumb finger                          | -1          | 1           | 0 (rad)      | 1.222 (rad) | robot0:A_THJ3                    | hinge | angle (rad) |
    | 17  | Horizontal angular position of the MCP joint of the thumb finger (adduction/abduction)  | -1          | 1           | -0.209 (rad) | 0.209(rad)  | robot0:A_THJ2                    | hinge | angle (rad) |
    | 18  | Vertical angular position of the MCP joint of the thumb finger (flexion/extension)      | -1          | 1           | -0.524 (rad) | 0.524 (rad) | robot0:A_THJ1                    | hinge | angle (rad) |
    | 19  | Angular position of the IP joint of the thumb finger (flexion/extension)                | -1          | 1           | -1.571 (rad) | 0 (rad)     | robot0:A_THJ0                    | hinge | angle (rad) |


    ### Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and egg states, as well as information about the goal.
    The dictionary consists of the following 3 keys:

    - `observation`: its value is an `ndarray` of shape `(61,)`. It consists of kinematic information of the egg object and finger joints. The elements of the array correspond to the
    following:

        | Num | Observation                                                       | Min    | Max    | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
        |-----|-------------------------------------------------------------------|--------|--------|----------------------------------------|----------|------------------------- |
        | 0   | Angular position of the horizontal wrist joint                    | -Inf   | Inf    | robot0:WRJ1                            | hinge    | angle (rad)              |
        | 1   | Angular position of the vertical wrist joint                      | -Inf   | Inf    | robot0:WRJ0                            | hinge    | angle (rad)              |
        | 2   | Horizontal angular position of the MCP joint of the forefinger    | -Inf   | Inf    | robot0:FFJ3                            | hinge    | angle (rad)              |
        | 3   | Vertical angular position of the MCP joint of the forefinge       | -Inf   | Inf    | robot0:FFJ2                            | hinge    | angle (rad)              |
        | 4   | Angular position of the PIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ1                            | hinge    | angle (rad)              |
        | 5   | Angular position of the DIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ0                            | hinge    | angle (rad)              |
        | 6   | Horizontal angular position of the MCP joint of the middle finger | -Inf   | Inf    | robot0:MFJ3                            | hinge    | angle (rad)              |
        | 7   | Vertical angular position of the MCP joint of the middle finger   | -Inf   | Inf    | robot0:MFJ2                            | hinge    | angle (rad)              |
        | 8   | Angular position of the PIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ1                            | hinge    | angle (rad)              |
        | 9   | Angular position of the DIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ0                            | hinge    | angle (rad)              |
        | 10  | Horizontal angular position of the MCP joint of the ring finger   | -Inf   | Inf    | robot0:RFJ3                            | hinge    | angle (rad)              |
        | 11  | Vertical angular position of the MCP joint of the ring finger     | -Inf   | Inf    | robot0:RFJ2                            | hinge    | angle (rad)              |
        | 12  | Angular position of the PIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ1                            | hinge    | angle (rad)              |
        | 13  | Angular position of the DIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ0                            | hinge    | angle (rad)              |
        | 14  | Angular position of the CMC joint of the little finger            | -Inf   | Inf    | robot0:LFJ4                            | hinge    | angle (rad)              |
        | 15  | Horizontal angular position of the MCP joint of the little finger | -Inf   | Inf    | robot0:LFJ3                            | hinge    | angle (rad)              |
        | 16  | Vertical angular position of the MCP joint of the little finger   | -Inf   | Inf    | robot0:LFJ2                            | hinge    | angle (rad)              |
        | 17  | Angular position of the PIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ1                            | hinge    | angle (rad)              |
        | 18  | Angular position of the DIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ0                            | hinge    | angle (rad)              |
        | 19  | Horizontal angular position of the CMC joint of the thumb finger  | -Inf   | Inf    | robot0:THJ4                            | hinge    | angle (rad)              |
        | 20  | Vertical Angular position of the CMC joint of the thumb finger    | -Inf   | Inf    | robot0:THJ3                            | hinge    | angle (rad)              |
        | 21  | Horizontal angular position of the MCP joint of the thumb finger  | -Inf   | Inf    | robot0:THJ2                            | hinge    | angle (rad)              |
        | 22  | Vertical angular position of the MCP joint of the thumb finger    | -Inf   | Inf    | robot0:THJ1                            | hinge    | angle (rad)              |
        | 23  | Angular position of the IP joint of the thumb finger              | -Inf   | Inf    | robot0:THJ0                            | hinge    | angle (rad)              |
        | 24  | Angular velocity of the horizontal wrist joint                    | -Inf   | Inf    | robot0:WRJ1                            | hinge    | angular velocity (rad/s) |
        | 25  | Angular velocity of the vertical wrist joint                      | -Inf   | Inf    | robot0:WRJ0                            | hinge    | angular velocity (rad/s) |
        | 26  | Horizontal angular velocity of the MCP joint of the forefinger    | -Inf   | Inf    | robot0:FFJ3                            | hinge    | angular velocity (rad/s) |
        | 27  | Vertical angular velocity of the MCP joint of the forefinge       | -Inf   | Inf    | robot0:FFJ2                            | hinge    | angular velocity (rad/s) |
        | 28  | Angular velocity of the PIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ1                            | hinge    | angular velocity (rad/s) |
        | 29  | Angular velocity of the DIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ0                            | hinge    | angular velocity (rad/s) |
        | 30  | Horizontal angular velocity of the MCP joint of the middle finger | -Inf   | Inf    | robot0:MFJ3                            | hinge    | angular velocity (rad/s) |
        | 31  | Vertical angular velocity of the MCP joint of the middle finger   | -Inf   | Inf    | robot0:MFJ2                            | hinge    | angular velocity (rad/s) |
        | 32  | Angular velocity of the PIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ1                            | hinge    | angular velocity (rad/s) |
        | 33  | Angular velocity of the DIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ0                            | hinge    | angular velocity (rad/s) |
        | 34  | Horizontal angular velocity of the MCP joint of the ring finger   | -Inf   | Inf    | robot0:RFJ3                            | hinge    | angular velocity (rad/s) |
        | 35  | Vertical angular velocity of the MCP joint of the ring finger     | -Inf   | Inf    | robot0:RFJ2                            | hinge    | angular velocity (rad/s) |
        | 36  | Angular velocity of the PIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ1                            | hinge    | angular velocity (rad/s) |
        | 37  | Angular velocity of the DIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ0                            | hinge    | angular velocity (rad/s) |
        | 38  | Angular velocity of the CMC joint of the little finger            | -Inf   | Inf    | robot0:LFJ4                            | hinge    | angular velocity (rad/s) |
        | 39  | Horizontal angular velocity of the MCP joint of the little finger | -Inf   | Inf    | robot0:LFJ3                            | hinge    | angular velocity (rad/s) |
        | 40  | Vertical angular velocity of the MCP joint of the little finger   | -Inf   | Inf    | robot0:LFJ2                            | hinge    | angular velocity (rad/s) |
        | 41  | Angular velocity of the PIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ1                            | hinge    | angular velocity (rad/s) |
        | 42  | Angular velocity of the DIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ0                            | hinge    | angular velocity (rad/s) |
        | 43  | Horizontal angular velocity of the CMC joint of the thumb finger  | -Inf   | Inf    | robot0:THJ4                            | hinge    | angular velocity (rad/s) |
        | 44  | Vertical Angular velocity of the CMC joint of the thumb finger    | -Inf   | Inf    | robot0:THJ3                            | hinge    | angular velocity (rad/s) |
        | 45  | Horizontal angular velocity of the MCP joint of the thumb finger  | -Inf   | Inf    | robot0:THJ2                            | hinge    | angular velocity (rad/s) |
        | 46  | Vertical angular position of the MCP joint of the thumb finger    | -Inf   | Inf    | robot0:THJ1                            | hinge    | angular velocity (rad/s) |
        | 47  | Angular velocity of the IP joint of the thumb finger              | -Inf   | Inf    | robot0:THJ0                            | hinge    | angular velocity (rad/s) |
        | 48  | Linear velocity of the egg in x direction                         | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 49  | Linear velocity of the egg in y direction                         | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 50  | Linear velocity of the egg in z direction                         | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 51  | Angular velocity of the egg in x axis                             | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 52  | Angular velocity of the egg in y axis                             | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 53  | Angular velocity of the egg in z axis                             |  -Inf  | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 54  | Position of the egg in the x coordinate                           | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 55  | Position of the egg in the y coordinate                           | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 56  | Position of the egg in the z coordinate                           | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 57  | w component of the quaternion orientation of the egg              | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 58  | x component of the quaternion orientation of the egg              | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 59  | y component of the quaternion orientation of the egg              | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 60  | z component of the quaternion orientation of the egg              | -Inf   | Inf    | object:joint                           | free     | -                        |

    - `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 7-dimensional `ndarray`, `(7,)`, that consists of the pose information of the egg.
    The elements of the array are the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
        | 0   | Target x coordinate of the egg                                                                                                        | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 1   | Target y coordinate of the egg                                                                                                        | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 2   | Target z coordinate of the egg                                                                                                        | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 3   | Target w component of the quaternion orientation of the egg                                                                           | -Inf   | Inf    | target:joint                           | free       | -            |
        | 4   | Target x component of the quaternion orientation of the egg                                                                           | -Inf   | Inf    | target:joint                           | free       | -            |
        | 5   | Target y component of the quaternion orientation of the egg                                                                           | -Inf   | Inf    | target:joint                           | free       | -            |
        | 6   | Target z component of the quaternion orientation of the egg                                                                           | -Inf   | Inf    | target:joint                           | free       | -            |


    - `achieved_goal`: this key represents the current state of the egg, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER).
    The value is an `ndarray` with shape `(7,)`. The elements of the array are the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
        | 0   | Current x coordinate of the egg                                                                                                       | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 1   | Current y coordinate of the egg                                                                                                       | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 2   | Current z coordinate of the egg                                                                                                       | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 3   | Current w component of the quaternion orientation of the egg                                                                          | -Inf   | Inf    | object:joint                           | free       | -            |
        | 4   | Current x component of the quaternion orientation of the egg                                                                          | -Inf   | Inf    | object:joint                           | free       | -            |
        | 5   | Current y component of the quaternion orientation of the egg                                                                          | -Inf   | Inf    | object:joint                           | free       | -            |
        | 6   | Current z component of the quaternion orientation of the egg                                                                          | -Inf   | Inf    | object:joint                           | free       | -            |


    ### Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the egg hasn't reached its final target pose, and `0` if the egg is in its final target pose. The egg is considered to have reached its final goal if the theta angle difference
    (theta angle of the [3D axis angle representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) is less than 0.1 and if the Euclidean distance to the target position is also less than 0.01 m.
    - *dense*: the returned reward is the negative summation of the Euclidean distance to the egg's target and the theta angle difference to the target orientation. The positional distance is multiplied by a factor of 10 to avoid being dominated
    by the rotational difference.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `HandManipulateEgg-v1`.
    However, for `dense` reward the id must be modified to `HandManipulateEggDense-v1` and initialized as follows:

    ```python
    import gymnasium as gym

    env = gym.make('HandManipulateEgg-v1')
    ```

    The rest of the id's of the other environment variations follow the same convention to select between a sparse or dense reward function.

    ### Starting State

    When the environment is reset the joints of the hand are initialized to their resting position with a 0 displacement. The egg's position and orientation are randomly selected. The initial position is set to `(x,y,z)=(1, 0.87, 0.2)` and an offset is added
    to each coordinate sampled from a normal distribution with 0 mean and 0.005 standard deviation.
    While the initial orientation is set to `(w,x,y,z)=(1,0,0,0)` and an axis is randomly selected depending on the environment variation to add an angle offset sampled from a uniform distribution with range `[-pi, pi]`.

    The target pose of the egg is obtained by adding a random offset to the initial egg pose. For the position the offset is sampled from a uniform distribution with range `[(x_min, x_max), (y_min,y_max), (z_min, z_max)] = [(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]`.
    The orientation offset is sampled from a uniform distribution with range `[-pi,pi]` and added to one of the Euler axis depending on the environment variation.


    ### Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ### Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

    ```
    import gymnasium as gym

    env = gym.make('HandManipulateEgg-v1', max_episode_steps=100)
    ```

    The same applies for the other environment variations.

    ### Version History

    * v1: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: the environment depends on `mujoco_py` which is no longer maintained.

    """

    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_EGG_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)


class MujocoPyHandEggEnv(MujocoPyManipulateEnv, EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoPyManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_EGG_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)


class MujocoHandPenEnv(MujocoManipulateEnv, EzPickle):
    """
    ### Description

    This environment was introduced in ["Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research"](https://arxiv.org/abs/1802.09464).

    The environment is based on the same robot hand as in the `HandReach` environment, the [Shadow Dexterous Hand](https://www.shadowrobot.com/). The task to be solved is
    very similar to that in the `HandManipulateBlock` environment, but in this case a pen is placed on the palm of the hand. The task is to then manipulate
    the pen such that a target pose is achieved. The goal is 7-dimensional and includes the target position (in Cartesian coordinates) and target rotation (in quaternions).
    In addition, variations of this environment can be used with increasing levels of difficulty:

    * `HandManipulatePenRotate-v1`: Random target rotation *x* and *y* axes of the pen and no target rotation around the *z* axis. No target position.
    * `HandManipulatePenFull-v1`:  Random target rotation x and y axes of the pen and no target rotation around the z axis. Random target position.

    ### Action Space

    The action space is a `Box(-1.0, 1.0, (20,), float32)`. The control actions are absolute angular positions of the actuated joints (non-coupled). The input of the control
    actions is set to a range between -1 and 1 by scaling the actual actuator angle ranges. The elements of the action array are the following:

    | Num | Action                                                                                  | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | --------------------------------------------------------------------------------------- | ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Angular position of the horizontal wrist joint (radial/ulnar deviation)                 | -1          | 1           | -0.489 (rad) | 0.14 (rad)  | robot0:A_WRJ1                    | hinge | angle (rad) |
    | 1   | Angular position of the horizontal wrist joint (flexion/extension)                      | -1          | 1           | -0.698 (rad) | 0.489 (rad) | robot0:A_WRJ0                    | hinge | angle (rad) |
    | 2   | Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)    | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_FFJ3                    | hinge | angle (rad) |
    | 3   | Vertical angular position of the MCP joint of the forefinger (flexion/extension)        | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_FFJ2                    | hinge | angle (rad) |
    | 4   | Angular position of the PIP joint of the forefinger (flexion/extension)                 | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_FFJ1                    | hinge | angle (rad) |
    | 5   | Horizontal angular position of the MCP joint of the middle finger (adduction/abduction) | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_MFJ3                    | hinge | angle (rad) |
    | 6   | Vertical angular position of the MCP joint of the middle finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_MFJ2                    | hinge | angle (rad) |
    | 7   | Angular position of the PIP joint of the middle finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_MFJ1                    | hinge | angle (rad) |
    | 8   | Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)   | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_RFJ3                    | hinge | angle (rad) |
    | 9   | Vertical angular position of the MCP joint of the ring finger (flexion/extension)       | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_RFJ2                    | hinge | angle (rad) |
    | 10  | Angular position of the PIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_RFJ1                    | hinge | angle (rad) |
    | 11  | Angular position of the CMC joint of the little finger                                  | -1          | 1           | 0 (rad)      | 0.785(rad)  | robot0:A_LFJ4                    | hinge | angle (rad) |
    | 12  | Horizontal angular position of the MCP joint of the little finger (adduction/abduction) | -1          | 1           | -0.349 (rad) | 0.349(rad)  | robot0:A_LFJ3                    | hinge | angle (rad) |
    | 13  | Vertical angular position of the MCP joint of the little finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_LFJ2                    | hinge | angle (rad) |
    | 14  | Angular position of the PIP joint of the little finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.571 (rad) | robot0:A_LFJ1                    | hinge | angle (rad) |
    | 15  | Horizontal angular position of the CMC joint of the thumb finger                        | -1          | 1           | -1.047 (rad) | 1.047 (rad) | robot0:A_THJ4                    | hinge | angle (rad) |
    | 16  | Vertical Angular position of the CMC joint of the thumb finger                          | -1          | 1           | 0 (rad)      | 1.222 (rad) | robot0:A_THJ3                    | hinge | angle (rad) |
    | 17  | Horizontal angular position of the MCP joint of the thumb finger (adduction/abduction)  | -1          | 1           | -0.209 (rad) | 0.209(rad)  | robot0:A_THJ2                    | hinge | angle (rad) |
    | 18  | Vertical angular position of the MCP joint of the thumb finger (flexion/extension)      | -1          | 1           | -0.524 (rad) | 0.524 (rad) | robot0:A_THJ1                    | hinge | angle (rad) |
    | 19  | Angular position of the IP joint of the thumb finger (flexion/extension)                | -1          | 1           | -1.571 (rad) | 0 (rad)     | robot0:A_THJ0                    | hinge | angle (rad) |


    ### Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and pen states, as well as information about the goal.
    The dictionary consists of the following 3 keys:

    - `observation`: its value is an `ndarray` of shape `(61,)`. It consists of kinematic information of the pen and finger joints. The elements of the array correspond to the
    following:

        | Num | Observation                                                       | Min    | Max    | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
        |-----|-------------------------------------------------------------------|--------|--------|----------------------------------------|----------|------------------------- |
        | 0   | Angular position of the horizontal wrist joint                    | -Inf   | Inf    | robot0:WRJ1                            | hinge    | angle (rad)              |
        | 1   | Angular position of the vertical wrist joint                      | -Inf   | Inf    | robot0:WRJ0                            | hinge    | angle (rad)              |
        | 2   | Horizontal angular position of the MCP joint of the forefinger    | -Inf   | Inf    | robot0:FFJ3                            | hinge    | angle (rad)              |
        | 3   | Vertical angular position of the MCP joint of the forefinge       | -Inf   | Inf    | robot0:FFJ2                            | hinge    | angle (rad)              |
        | 4   | Angular position of the PIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ1                            | hinge    | angle (rad)              |
        | 5   | Angular position of the DIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ0                            | hinge    | angle (rad)              |
        | 6   | Horizontal angular position of the MCP joint of the middle finger | -Inf   | Inf    | robot0:MFJ3                            | hinge    | angle (rad)              |
        | 7   | Vertical angular position of the MCP joint of the middle finger   | -Inf   | Inf    | robot0:MFJ2                            | hinge    | angle (rad)              |
        | 8   | Angular position of the PIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ1                            | hinge    | angle (rad)              |
        | 9   | Angular position of the DIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ0                            | hinge    | angle (rad)              |
        | 10  | Horizontal angular position of the MCP joint of the ring finger   | -Inf   | Inf    | robot0:RFJ3                            | hinge    | angle (rad)              |
        | 11  | Vertical angular position of the MCP joint of the ring finger     | -Inf   | Inf    | robot0:RFJ2                            | hinge    | angle (rad)              |
        | 12  | Angular position of the PIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ1                            | hinge    | angle (rad)              |
        | 13  | Angular position of the DIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ0                            | hinge    | angle (rad)              |
        | 14  | Angular position of the CMC joint of the little finger            | -Inf   | Inf    | robot0:LFJ4                            | hinge    | angle (rad)              |
        | 15  | Horizontal angular position of the MCP joint of the little finger | -Inf   | Inf    | robot0:LFJ3                            | hinge    | angle (rad)              |
        | 16  | Vertical angular position of the MCP joint of the little finger   | -Inf   | Inf    | robot0:LFJ2                            | hinge    | angle (rad)              |
        | 17  | Angular position of the PIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ1                            | hinge    | angle (rad)              |
        | 18  | Angular position of the DIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ0                            | hinge    | angle (rad)              |
        | 19  | Horizontal angular position of the CMC joint of the thumb finger  | -Inf   | Inf    | robot0:THJ4                            | hinge    | angle (rad)              |
        | 20  | Vertical Angular position of the CMC joint of the thumb finger    | -Inf   | Inf    | robot0:THJ3                            | hinge    | angle (rad)              |
        | 21  | Horizontal angular position of the MCP joint of the thumb finger  | -Inf   | Inf    | robot0:THJ2                            | hinge    | angle (rad)              |
        | 22  | Vertical angular position of the MCP joint of the thumb finger    | -Inf   | Inf    | robot0:THJ1                            | hinge    | angle (rad)              |
        | 23  | Angular position of the IP joint of the thumb finger              | -Inf   | Inf    | robot0:THJ0                            | hinge    | angle (rad)              |
        | 24  | Angular velocity of the horizontal wrist joint                    | -Inf   | Inf    | robot0:WRJ1                            | hinge    | angular velocity (rad/s) |
        | 25  | Angular velocity of the vertical wrist joint                      | -Inf   | Inf    | robot0:WRJ0                            | hinge    | angular velocity (rad/s) |
        | 26  | Horizontal angular velocity of the MCP joint of the forefinger    | -Inf   | Inf    | robot0:FFJ3                            | hinge    | angular velocity (rad/s) |
        | 27  | Vertical angular velocity of the MCP joint of the forefinge       | -Inf   | Inf    | robot0:FFJ2                            | hinge    | angular velocity (rad/s) |
        | 28  | Angular velocity of the PIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ1                            | hinge    | angular velocity (rad/s) |
        | 29  | Angular velocity of the DIP joint of the forefinger               | -Inf   | Inf    | robot0:FFJ0                            | hinge    | angular velocity (rad/s) |
        | 30  | Horizontal angular velocity of the MCP joint of the middle finger | -Inf   | Inf    | robot0:MFJ3                            | hinge    | angular velocity (rad/s) |
        | 31  | Vertical angular velocity of the MCP joint of the middle finger   | -Inf   | Inf    | robot0:MFJ2                            | hinge    | angular velocity (rad/s) |
        | 32  | Angular velocity of the PIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ1                            | hinge    | angular velocity (rad/s) |
        | 33  | Angular velocity of the DIP joint of the middle finger            | -Inf   | Inf    | robot0:MFJ0                            | hinge    | angular velocity (rad/s) |
        | 34  | Horizontal angular velocity of the MCP joint of the ring finger   | -Inf   | Inf    | robot0:RFJ3                            | hinge    | angular velocity (rad/s) |
        | 35  | Vertical angular velocity of the MCP joint of the ring finger     | -Inf   | Inf    | robot0:RFJ2                            | hinge    | angular velocity (rad/s) |
        | 36  | Angular velocity of the PIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ1                            | hinge    | angular velocity (rad/s) |
        | 37  | Angular velocity of the DIP joint of the ring finger              | -Inf   | Inf    | robot0:RFJ0                            | hinge    | angular velocity (rad/s) |
        | 38  | Angular velocity of the CMC joint of the little finger            | -Inf   | Inf    | robot0:LFJ4                            | hinge    | angular velocity (rad/s) |
        | 39  | Horizontal angular velocity of the MCP joint of the little finger | -Inf   | Inf    | robot0:LFJ3                            | hinge    | angular velocity (rad/s) |
        | 40  | Vertical angular velocity of the MCP joint of the little finger   | -Inf   | Inf    | robot0:LFJ2                            | hinge    | angular velocity (rad/s) |
        | 41  | Angular velocity of the PIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ1                            | hinge    | angular velocity (rad/s) |
        | 42  | Angular velocity of the DIP joint of the little finger            | -Inf   | Inf    | robot0:LFJ0                            | hinge    | angular velocity (rad/s) |
        | 43  | Horizontal angular velocity of the CMC joint of the thumb finger  | -Inf   | Inf    | robot0:THJ4                            | hinge    | angular velocity (rad/s) |
        | 44  | Vertical Angular velocity of the CMC joint of the thumb finger    | -Inf   | Inf    | robot0:THJ3                            | hinge    | angular velocity (rad/s) |
        | 45  | Horizontal angular velocity of the MCP joint of the thumb finger  | -Inf   | Inf    | robot0:THJ2                            | hinge    | angular velocity (rad/s) |
        | 46  | Vertical angular position of the MCP joint of the thumb finger    | -Inf   | Inf    | robot0:THJ1                            | hinge    | angular velocity (rad/s) |
        | 47  | Angular velocity of the IP joint of the thumb finger              | -Inf   | Inf    | robot0:THJ0                            | hinge    | angular velocity (rad/s) |
        | 48  | Linear velocity of the pen in x direction                         | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 49  | Linear velocity of the pen in y direction                         | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 50  | Linear velocity of the pen in z direction                         | -Inf   | Inf    | object:joint                           | free     | velocity (m/s)           |
        | 51  | Angular velocity of the pen in x axis                             | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 52  | Angular velocity of the pen in y axis                             | -Inf   | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 53  | Angular velocity of the pen in z axis                             |  -Inf  | Inf    | object:joint                           | free     | angular velocity (rad/s) |
        | 54  | Position of the pen in the x coordinate                           | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 55  | Position of the pen in the y coordinate                           | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 56  | Position of the pen in the z coordinate                           | -Inf   | Inf    | object:joint                           | free     | position (m)             |
        | 57  | w component of the quaternion orientation of the pen              | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 58  | x component of the quaternion orientation of the pen              | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 59  | y component of the quaternion orientation of the pen              | -Inf   | Inf    | object:joint                           | free     | -                        |
        | 60  | z component of the quaternion orientation of the pen              | -Inf   | Inf    | object:joint                           | free     | -                        |

    - `desired_goal`: this key represents the final goal to be achieved. In this environment it is a 7-dimensional `ndarray`, `(7,)`, that consists of the pose information of the pen.
    The elements of the array are the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
        | 0   | Target x coordinate of the pen                                                                                                        | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 1   | Target y coordinate of the pen                                                                                                        | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 2   | Target z coordinate of the pen                                                                                                        | -Inf   | Inf    | target:joint                           | free       | position (m) |
        | 3   | Target w component of the quaternion orientation of the pen                                                                           | -Inf   | Inf    | target:joint                           | free       | -            |
        | 4   | Target x component of the quaternion orientation of the pen                                                                           | -Inf   | Inf    | target:joint                           | free       | -            |
        | 5   | Target y component of the quaternion orientation of the pen                                                                           | -Inf   | Inf    | target:joint                           | free       | -            |
        | 6   | Target z component of the quaternion orientation of the pen                                                                           | -Inf   | Inf    | target:joint                           | free       | -            |


    - `achieved_goal`: this key represents the current state of the pen, as if it would have achieved a goal. This is useful for goal orientated learning algorithms such as those that use [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (HER).
    The value is an `ndarray` with shape `(7,)`. The elements of the array are the following:

        | Num | Observation                                                                                                                           | Min    | Max    | Joint Name (in corresponding XML file) | Joint Type | Unit         |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|----------------------------------------|------------|--------------|
        | 0   | Current x coordinate of the pen                                                                                                       | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 1   | Current y coordinate of the pen                                                                                                       | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 2   | Current z coordinate of the pen                                                                                                       | -Inf   | Inf    | object:joint                           | free       | position (m) |
        | 3   | Current w component of the quaternion orientation of the pen                                                                          | -Inf   | Inf    | object:joint                           | free       | -            |
        | 4   | Current x component of the quaternion orientation of the pen                                                                          | -Inf   | Inf    | object:joint                           | free       | -            |
        | 5   | Current y component of the quaternion orientation of the pen                                                                          | -Inf   | Inf    | object:joint                           | free       | -            |
        | 6   | Current z component of the quaternion orientation of the pen                                                                          | -Inf   | Inf    | object:joint                           | free       | -            |


    ### Rewards
    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the pen hasn't reached its final target pose, and `0` if the pen is in its final target pose. The pen is considered to have reached its final goal if the theta angle difference
    (theta angle of the [3D axis angle representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) is less than 0.1 and if the Euclidean distance to the target position is also less than 0.01 m.
    - *dense*: the returned reward is the negative summation of the Euclidean distance to the pen's target and the theta angle difference to the target orientation. The positional distance is multiplied by a factor of 10 to avoid being dominated
    by the rotational difference.

    To initialize this environment with one of the mentioned reward functions the type of reward must be specified in the id string when the environment is initialized. For `sparse` reward the id is the default of the environment, `HandManipulatePen-v1`.
    However, for `dense` reward the id must be modified to `HandManipulatePenDense-v1` and initialized as follows:

    ```python
    import gymnasium as gym

    env = gym.make('HandManipulatePen-v1')
    ```

    The rest of the id's of the other environment variations follow the same convention to select between a sparse or dense reward function.

    ### Starting State

    When the environment is reset the joints of the hand are initialized to their resting position with a 0 displacement. The pen's position and orientation are randomly selected. The initial position is set to `(x,y,z)=(1, 0.87, 0.2)` and an offset is added
    to each coordinate sampled from a normal distribution with 0 mean and 0.005 standard deviation.
    While the initial orientation is set to `(w,x,y,z)=(1,0,0,0)` and an axis is randomly selected depending on the environment variation to add an angle offset sampled from a uniform distribution with range `[-pi, pi]`.

    The target pose of the pen is obtained by adding a random offset to the initial pen pose. For the position the offset is sampled from a uniform distribution with range `[(x_min, x_max), (y_min,y_max), (z_min, z_max)] = [(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]`.
    The orientation offset is sampled from a uniform distribution with range `[-pi,pi]` and added to one of the Euler axis depending on the environment variation.


    ### Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ### Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

    ```python
    import gymnasium as gym

    env = gym.make('HandManipulatePen-v1', max_episode_steps=100)
    ```

    The same applies for the other environment variations.

    ### Version History

    * v1: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: the environment depends on `mujoco_py` which is no longer maintained.

    """

    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_PEN_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False,
            reward_type=reward_type,
            ignore_z_target_rotation=True,
            distance_threshold=0.05,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)


class MujocoPyHandPenEnv(MujocoPyManipulateEnv, EzPickle):
    def __init__(
        self,
        target_position="random",
        target_rotation="xyz",
        reward_type="sparse",
        **kwargs,
    ):
        MujocoPyManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_PEN_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            randomize_initial_rotation=False,
            reward_type=reward_type,
            ignore_z_target_rotation=True,
            distance_threshold=0.05,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)
