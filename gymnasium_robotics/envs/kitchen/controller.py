import numpy as np
from gymnasium_robotics.utils.controller_utils import RingBuffer
from collections.abc import Iterable


import numpy as np

from gymnasium_robotics.utils.rotations import mat2quat, mat2euler, euler2mat, quat2mat, quat_slerp

class LinearInterpolator:
    """
    Simple class for implementing a linear interpolator.
    Abstracted to interpolate n-dimensions
    Args:
        ndim (int): Number of dimensions to interpolate
        controller_freq (float): Frequency (Hz) of the controller
        policy_freq (float): Frequency (Hz) of the policy model
        ramp_ratio (float): Percentage of interpolation timesteps across which we will interpolate to a goal position.
            :Note: Num total interpolation steps will be equal to np.floor(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update
        ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
            Specified string determines assumed type of input:
                `'euler'`: Euler orientation inputs
                `'quat'`: Quaternion inputs
    """

    def __init__(
        self,
        ndim,
        controller_freq,
        policy_freq,
        ramp_ratio=0.2,
        use_delta_goal=False,
        ori_interpolate=None,
    ):
        self.dim = ndim  # Number of dimensions to interpolate
        self.ori_interpolate = ori_interpolate  # Whether this is interpolating orientation or not
        self.order = 1  # Order of the interpolator (1 = linear)
        self.step = 0  # Current step of the interpolator
        self.total_steps = np.ceil(
            ramp_ratio * controller_freq / policy_freq
        )  # Total num steps per interpolator action
        self.use_delta_goal = use_delta_goal  # Whether to use delta or absolute goals (currently
        # not implemented yet- TODO)
        self.set_states(dim=ndim, ori=ori_interpolate)

    def set_states(self, dim=None, ori=None):
        """
        Updates self.dim and self.ori_interpolate.
        Initializes self.start and self.goal with correct dimensions.
        Args:
            ndim (None or int): Number of dimensions to interpolate
            ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
                Specified string determines assumed type of input:
                    `'euler'`: Euler orientation inputs
                    `'quat'`: Quaternion inputs
        """
        # Update self.dim and self.ori_interpolate
        self.dim = dim if dim is not None else self.dim
        self.ori_interpolate = ori if ori is not None else self.ori_interpolate

        # Set start and goal states
        if self.ori_interpolate is not None:
            if self.ori_interpolate == "euler":
                self.start = np.zeros(3)
            else:  # quaternions
                self.start = np.array((0, 0, 0, 1))
        else:
            self.start = np.zeros(self.dim)
        self.goal = np.array(self.start)

    def set_goal(self, goal):
        """
        Takes a requested (absolute) goal and updates internal parameters for next interpolation step
        Args:
            np.array: Requested goal (absolute value). Should be same dimension as self.dim
        """
        # First, check to make sure requested goal shape is the same as self.dim
        if goal.shape[0] != self.dim:
            print("Requested goal: {}".format(goal))
            raise ValueError(
                "LinearInterpolator: Input size wrong for goal; got {}, needs to be {}!".format(goal.shape[0], self.dim)
            )

        # Update start and goal
        self.start = np.array(self.goal)
        self.goal = np.array(goal)

        # Reset interpolation steps
        self.step = 0

    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.
        NOTE: If this interpolator is for orientation, it is assumed to be receiving either euler angles or quaternions
        Returns:
            np.array: Next position in the interpolated trajectory
        """
        # Grab start position
        x = np.array(self.start)
        # Calculate the desired next step based on remaining interpolation steps
        if self.ori_interpolate is not None:
            # This is an orientation interpolation, so we interpolate linearly around a sphere instead
            goal = np.array(self.goal)
            if self.ori_interpolate == "euler":
                # this is assumed to be euler angles (x,y,z), so we need to first map to quat
                x = mat2quat(euler2mat(x))
                goal = mat2quat(euler2mat(self.goal))

            # Interpolate to the next sequence
            x_current = quat_slerp(x, goal, fraction=(self.step + 1) / self.total_steps)
            if self.ori_interpolate == "euler":
                # Map back to euler
                x_current = mat2euler(quat2mat(x_current))
        else:
            # This is a normal interpolation
            dx = (self.goal - x) / (self.total_steps - self.step)
            x_current = x + dx

        # Increment step if there's still steps remaining based on ramp ratio
        if self.step < self.total_steps - 1:
            self.step += 1

        # Return the new interpolated step
        return x_current

class JointVelocityController:
    """
    Controller for controlling the robot arm's joint velocities. This is simply a P controller with desired torques
    (pre gravity compensation) taken to be proportional to the velocity error of the robot joints.
    NOTE: Control input actions assumed to be taken as absolute joint velocities. A given action to this
    controller is assumed to be of the form: (vel_j0, vel_j1, ... , vel_jn-1) for an n-joint robot
    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from
        eef_name (str): Name of controlled robot arm's end effector (from robot XML)
        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:
            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities
        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range
        input_max (float or list of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller
        input_min (float or list of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller
        output_max (float or list of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller
        output_min (float or list of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller
        kp (float or list of float): velocity gain for determining desired torques based upon the joint vel errors.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)
        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller
        velocity_limits (2-list of float or 2-list of list of floats): Limits (m/s) below and above which the magnitude
            of a calculated goal joint velocity will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)
        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint velocities
            to the goal joint velocities during each timestep between inputted actions
        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    """

    def __init__(
        self,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=0.5,
        output_min=-0.5,
        kp=0.3,
        policy_freq=20,
        control_freq=500,
        velocity_limits=None,
    ):

        # Control dimension
        self.control_dim = 7

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # gains and corresopnding vars
        self.kp = self.nums2array(kp, self.control_dim)
        # if kp is a single value, map wrist gains accordingly (scale down x10 for final two joints)

        if type(kp) is float or type(kp) is int:
            # Scale kpp according to how wide the actuator range is for this robot
            self.actuator_min, self.actuator_max = actuator_range
            self.kp = kp * (self.actuator_max - self.actuator_min)
        self.ki = self.kp * 0.005
        self.kd = self.kp * 0.001
        self.last_err = np.zeros(self.control_dim)
        self.derr_buf = RingBuffer(dim=self.control_dim, length=5)
        self.summed_err = np.zeros(self.control_dim)
        self.saturated = False
        self.last_joint_vel = np.zeros(self.control_dim)

        # limits
        self.velocity_limits = np.array(velocity_limits) if velocity_limits is not None else None

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = LinearInterpolator(ndim=self.control_dim, controller_freq=control_freq, policy_freq=policy_freq)

        # initialize torques and goal velocity
        self.goal_vel = None  # Goal velocity desired, pre-compensation
        self.current_vel = np.zeros(self.control_dim)  # Current velocity setpoint, pre-compensation
        self.torques = None  # Torques returned every time run_controller is called
    
    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array
        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force
        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    def set_goal(self, velocities):
        """
        Sets goal based on input @velocities.
        Args:
            velocities (Iterable): Desired joint velocities
        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()

        # Otherwise, check to make sure velocities is size self.joint_dim
        assert (
            len(velocities) == self.control_dim
        ), "Goal action must be equal to the robot's joint dimension space! Expected {}, got {}".format(
            self.control_dim, len(velocities)
        )

        self.goal_vel = self.scale_action(velocities)
        if self.velocity_limits is not None:
            self.goal_vel = np.clip(self.goal_vel, self.velocity_limits[0], self.velocity_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_vel)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint
        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        if self.goal_vel is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        self.current_vel = self.interpolator.get_interpolated_goal()

        # Compute necessary error terms for PID velocity controller
        err = self.current_vel - self.joint_vel
        derr = err - self.last_err
        self.last_err = err
        self.derr_buf.push(derr)

        # Only add to I component if we're not saturated (anti-windup)
        if not self.saturated:
            self.summed_err += err

        # Compute command torques via PID velocity controller plus gravity compensation torques
        torques = self.kp * err + self.ki * self.summed_err + self.kd * self.derr_buf.average + self.torque_compensation

        # Clip torques
        self.torques = self.clip_torques(torques)

        # Check if we're saturated
        self.saturated = False if np.sum(np.abs(self.torques - torques)) == 0 else True

        self.new_update = True

        # Return final torques
        return self.torques

    def reset_goal(self):
        """
        Resets joint velocity goal to be all zeros
        """
        self.goal_vel = np.zeros(self.control_dim)

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_vel)
    
    def update(self):
        """
        Updates the state of the robot arm, including end effector pose / orientation / velocity, joint pos/vel,
        jacobian, and mass matrix. By default, since this is a non-negligible computation, multiple redundant calls
        will be ignored via the self.new_update attribute flag. However, if the @force flag is set, the update will
        occur regardless of that state of self.new_update. This base class method of @run_controller resets the
        self.new_update flag

        Args:
            force (bool): Whether to force an update to occur or not
        """

        # Only run update if self.new_update or force flag is set
        if self.new_update:
            self.joint_vel = np.array(self.data.qvel[self.qvel_index])
            # Clear self.new_update
            self.new_update = False

    @property
    def torque_compensation(self):
        """
        Gravity compensation for this robot arm

        Returns:
            np.array: torques
        """
        return self.data.qfrc_bias[self.qvel_index]
    
    def clip_torques(self, torques):
        """
        Clips the torques to be within the actuator limits

        Args:
            torques (Iterable): Torques to clip

        Returns:
            np.array: Clipped torques
        """
        return np.clip(torques, self.actuator_min, self.actuator_max)
