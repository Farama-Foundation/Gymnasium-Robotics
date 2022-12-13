import mujoco
import numpy as np
from gymnasium_robotics.envs.kitchen.controller import IKController
from gymnasium_robotics.utils.rotations import euler2quat
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer

if __name__ == "__main__":
    model_path = "../assets/kitchen_franka/franka_assets/franka_panda.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    eef_id = model.site('EEF').id
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    current_eef_quat = np.empty(4)
    init_eef_pos = data.site_xpos[eef_id].copy()
    init_eef_pos[2] -= 0.4
    # init_eef_pos[0] += 0.2
    controller = IKController(model, data)
    
    renderer = WindowViewer(model, data)
    
    xyz_rotation = np.zeros(3)
    xyz_rotation[0] = 0.5
    # xyz_rotation[1] = 0.3
    quat_rot = euler2quat(xyz_rotation)
    target_rot = np.empty(4)
    mujoco.mju_mat2Quat(current_eef_quat, data.site_xmat[eef_id].copy())
    mujoco.mju_mulQuat(target_rot, quat_rot, current_eef_quat)
    for i in range(100000):
        
        delta_qpos = controller.compute_qpos(init_eef_pos, target_rot)
        
        data.ctrl[:] += delta_qpos
        mujoco.mj_step(model, data, nstep=50)
                  
        renderer.render()
