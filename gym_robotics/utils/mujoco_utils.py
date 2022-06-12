import site
import numpy as np

from gym import error

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )

MJ_OBJ_TYPES = [
    "mjOBJ_BODY",
    "mjOBJ_JOINT",
    "mjOBJ_GEOM",
    "mjOBJ_SITE",
    "mjOBJ_LIGHT",
    "mjOBJ_CAMERA",
    "mjOBJ_ACTUATOR",
    "mjOBJ_SENSOR",
    "mjOBJ_TENDON",
    "mjOBJ_MESH",
]


def extract_mj_names(model, obj_type):

    if obj_type == mujoco.mjtObj.mjOBJ_BODY:
        name_addr = model.name_bodyadr
        n_obj = model.nbody

    elif obj_type == mujoco.mjtObj.mjOBJ_JOINT:
        name_addr = model.name_jntadr
        n_obj = model.njnt

    elif obj_type == mujoco.mjtObj.mjOBJ_GEOM:
        name_addr = model.name_geomadr
        n_obj = model.ngeom

    elif obj_type == mujoco.mjtObj.mjOBJ_SITE:
        name_addr = model.name_siteadr
        n_obj = model.nsite

    elif obj_type == mujoco.mjtObj.mjOBJ_LIGHT:
        name_addr = model.name_lightadr
        n_obj = model.nlight

    elif obj_type == mujoco.mjtObj.mjOBJ_CAMERA:
        name_addr = model.name_camadr
        n_obj = model.ncam

    elif obj_type == mujoco.mjtObj.mjOBJ_ACTUATOR:
        name_addr = model.name_actuatoradr
        n_obj = model.nu

    elif obj_type == mujoco.mjtObj.mjOBJ_SENSOR:
        name_addr = model.name_sensoradr
        n_obj = model.nsensor

    elif obj_type == mujoco.mjtObj.mjOBJ_TENDON:
        name_addr = model.name_tendonadr
        n_obj = model.ntendon

    elif obj_type == mujoco.mjtObj.mjOBJ_MESH:
        name_addr = model.name_meshadr
        n_obj = model.nmesh
    else:
        raise error.ValueError("")

    id2name = {i: None for i in range(n_obj)}
    name2id = {}
    for addr in name_addr:
        name = model.names[addr:].split(b"\x00")[0].decode()
        if name:
            obj_id = mujoco.mj_name2id(model, obj_type, name)
            assert 0 <= obj_id < n_obj and id2name[obj_id] is None
            name2id[name] = obj_id
            id2name[obj_id] = name

    return tuple(id2name[id] for id in sorted(name2id.values())), name2id, id2name


def robot_get_obs(model, data, joint_names):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if data.qpos is not None and joint_names:
        names = [n for n in joint_names if n.startswith("robot")]
        return (
            np.squeeze(np.array([get_joint_qpos(model, data, name) for name in names])),
            np.squeeze(np.array([get_joint_qvel(model, data, name) for name in names])),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(model, data, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if model.nmocap > 0:
        _, action = np.split(action, (model.nmocap * 7,))

    if len(data.ctrl) > 0:
        for i in range(action.shape[0]):
            if model.actuator_biastype[i] == 0:
                data.ctrl[i] = action[i]
            else:
                idx = model.jnt_qposadr[model.actuator_trnid[i, 0]]
                data.ctrl[i] = data.qpos[idx] + action[i]


def mocap_set_action(model, data, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if model.nmocap > 0:
        action, _ = np.split(action, (model.nmocap * 7,))
        action = action.reshape(model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(model, data)
        data.mocap_pos[:] = data.mocap_pos + pos_delta
        data.mocap_quat[:] = data.mocap_quat + quat_delta


def reset_mocap_welds(model, data):
    """Resets the mocap welds that we use for actuation."""
    if model.nmocap > 0 and model.eq_data is not None:
        for i in range(model.eq_data.shape[0]):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                model.eq_data[i, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)


def reset_mocap2body_xpos(model, data):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if model.eq_type is None or model.eq_obj1id is None or model.eq_obj2id is None:
        return
    for eq_type, obj1_id, obj2_id in zip(
        model.eq_type, model.eq_obj1id, model.eq_obj2id
    ):
        if eq_type != mujoco.mjtEq.mjEQ_WELD:
            continue

        mocap_id = model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        data.mocap_pos[mocap_id][:] = data.xpos[body_idx]
        data.mocap_quat[mocap_id][:] = data.xquat[body_idx]


def get_site_jacp(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    return jacp


def get_site_jacr(model, data, site_id):
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, None, jacr, site_id)

    return jacr


def set_joint_qpos(model, data, name, value):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (
            end_idx - start_idx
        ), "Value has incorrect shape %s: %s" % (name, value)
    data.qpos[start_idx:end_idx] = value


def set_joint_qvel(model, data, name, value):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_dofadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 3
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim
    value = np.array(value)
    if ndim > 1:
        assert value.shape == (
            end_idx - start_idx
        ), "Value has incorrect shape %s: %s" % (name, value)
    data.qvel[start_idx:end_idx] = value


def get_joint_qpos(model, data, name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qpos[start_idx:end_idx]


def get_joint_qvel(model, data, name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    start_idx = joint_addr
    end_idx = joint_addr + ndim

    return data.qvel[start_idx:end_idx]


def get_site_xpos(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xpos[site_id]


def get_site_xvelp(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    jacp = get_site_jacp(model, data, site_id)
    xvelp = jacp @ data.qvel
    return xvelp


def get_site_xvelr(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    jacp = get_site_jacr(model, data, site_id)
    xvelp = jacp @ data.qvel
    return xvelp


def set_mocap_pos(model, data, name, value):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mocap_id = model.body_mocapid[body_id]
    data.mocap_pos[mocap_id] = value


def set_mocap_quat(model, data, name, value):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    mocap_id = model.body_mocapid[body_id]
    data.mocap_quat[mocap_id] = value


def get_site_xmat(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return data.site_xmat[site_id].reshape(3, 3)


class MujocoModelNames(object):
    def __init__(self, model):
        (
            self._body_names,
            self._body_name2id,
            self._body_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_BODY)
        (
            self._joint_names,
            self._joint_name2id,
            self._joint_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_JOINT)
        (
            self._geom_names,
            self._geom_name2id,
            self._geom_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_GEOM)
        (
            self._site_names,
            self._site_name2id,
            self._site_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_SITE)
        (
            self._camera_names,
            self._camera_name2id,
            self._camera_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_CAMERA)
        (
            self._actuator_names,
            self._actuator_name2id,
            self._actuator_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR)
        (
            self._sensor_names,
            self._sensor_name2id,
            self._sensor_id2name,
        ) = extract_mj_names(model, mujoco.mjtObj.mjOBJ_SENSOR)

    @property
    def body_names(self):
        return self._body_names

    @property
    def body_name2id(self):
        return self._body_name2id

    @property
    def body_id2name(self):
        return self._body_id2name

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def joint_name2id(self):
        return self._joint_name2id

    @property
    def joint_id2name(self):
        return self._joint_id2name

    @property
    def geom_names(self):
        return self._geom_names

    @property
    def geom_name2id(self):
        return self._geom_name2id

    @property
    def geom_id2name(self):
        return self._geom_id2name

    @property
    def site_names(self):
        return self._site_names

    @property
    def site_name2id(self):
        return self._site_name2id

    @property
    def site_id2name(self):
        return self._site_id2name

    @property
    def camera_names(self):
        return self._camera_names

    @property
    def camera_name2id(self):
        return self._camera_name2id

    @property
    def camera_id2name(self):
        return self._camera_id2name

    @property
    def actuator_names(self):
        return self._actuator_names

    @property
    def actuator_name2id(self):
        return self._actuator_name2id

    @property
    def actuator_id2name(self):
        return self._actuator_id2name

    @property
    def sensor_names(self):
        return self._sensor_names

    @property
    def sensor_name2id(self):
        return self._sensor_name2id

    @property
    def sensor_id2name(self):
        return self._sensor_id2name
