import mujoco
import numpy as np


class IKController:
    def __init__(self, model, data, regularization_strength: float = 0.3):
        """Inverse Kinematic solver.

         The input of the controller are a target cartesian position and quaternion orientation for
         the frame of the end-effector.
         The controller will output angular displacements in the joint space to achieve the desired
         end-effector consiguration.

         This controller uses Damped Least Squares (DLS) to iteratively approximate to the solution and avoid singularities.
         More information can be read in the following paper. [Introduction to Inverse Kinematics with Jacobian Transpose,
         Pseudoinverse and Damped Least Squares methods](https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf)

        The controller implementation has also been inspired by the following repositories:
        [dexterity](https://github.com/kevinzakka/dexterity/tree/main/dexterity/inverse_kinematics), and
        [dm_robotics](https://github.com/deepmind/dm_robotics/blob/main/py/moma/utils/ik_solver.py). Even though these
        implementations use an integration technique to compute joint velocities instead of joint position displacements.

         Args:
             model (MjModel): mujoco model structure.
             data (MjData): mujoco data structure.
             regularization_strength (float): regularization parameter for DLS
        """
        self.model = model
        self.data = data

        self.regularization_strength = regularization_strength
        # End effector frame to compute IK
        self.eef_id = self.model.site("EEF").id

    def compute_qpos_delta(self, target_pos, target_quat):
        """Return joint position displacement to achieve a target end-effector pose.

        Args:
            target_pos (np.ndarray): target end-effector position
            target_quat (np.ndarray): target end-effector quaternion orientation

        Returns:
            qpos_increase (np.ndarray): joint displacement order by its mujoco id
        """
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        err = np.empty(6)
        err_pos, err_rot = err[:3], err[3:]
        eef_xquat = np.empty(4)
        neg_eef_xquat = np.empty(4)
        err_rot_quat = np.empty(4)

        eef_xpos = self.data.site_xpos[self.eef_id]
        eef_xmat = self.data.site_xmat[self.eef_id]

        # Translation error
        err_pos[:] = target_pos - eef_xpos

        # Rotation error
        mujoco.mju_mat2Quat(eef_xquat, eef_xmat)
        mujoco.mju_negQuat(neg_eef_xquat, eef_xquat)
        mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_eef_xquat)
        mujoco.mju_quat2Vel(err_rot, err_rot_quat, 50)
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.eef_id)
        jac = np.concatenate((jac_pos, jac_rot), axis=0)
        qpos_increase = self.solve_DLS(jac, err)

        return qpos_increase

    def solve_DLS(self, jac_joints, error):
        """Computes the Least Mean Squares algorithm over the following equation.

        Where `e` is the end-effector error, J is the joint space Jacobian,
        `Δθ` is the joint displacement to solve, and `τ` is the regularization factor.

        .. math::

            J^{T}JΔθ=J^{T}e + Iτ

        Args:
            jac_joints (np.ndarray): system joint space Jacobian
            error (np.ndarray): end-effector target error

        Returns:
            Δqpos (np.ndarray): joint position displacement
        """
        hess_approx = jac_joints.T.dot(jac_joints)
        hess_approx += np.eye(hess_approx.shape[0]) * self.regularization_strength
        joint_delta = jac_joints.T.dot(error)
        # Least-squares solution
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
