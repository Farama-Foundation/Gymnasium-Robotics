import mujoco
import numpy as np


class IKController:
    def __init__(self, model, data, max_steps=100):
        self.model = model
        self.data = data
        self.max_steps = max_steps

        self.eef_id = self.model.site("EEF").id

        self.init_pos = self.data.site_xpos
        self.init_rot = self.data.site_xmat[self.eef_id].reshape(3, 3)

    def compute_qpos(self, target_pos, target_quat):

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
        # err_rot *= self.model.opt.timestep
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.eef_id)
        jac = np.concatenate((jac_pos, jac_rot), axis=0)
        qpos_increase = null_space_method(jac, err)

        return qpos_increase


def null_space_method(jac_joints, delta, regularization_strength=0.3):
    hess_approx = jac_joints.T.dot(jac_joints)
    hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
    joint_delta = jac_joints.T.dot(delta)
    # Least-squares solution
    return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
