import gymnasium as gym
import numpy as np
import pytest

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


@pytest.mark.parametrize("env_id", ["AdroitHandHammer-v2", "AdroitHandHammerSparse-v2"])
def test_set_env_state_accepts_full_hammer_state(env_id):
    env = gym.make(env_id, disable_env_checker=True)
    adroit_env = env.unwrapped

    try:
        adroit_env.reset(seed=123)

        qpos = adroit_env.data.qpos.copy()
        qvel = adroit_env.data.qvel.copy()
        board_pos = np.array([0.02, -0.03, 0.18])
        nail_qpos_addr = adroit_env.model.jnt_qposadr[
            adroit_env._model_names.joint_name2id["nail_dir"]
        ]

        qpos[nail_qpos_addr] = 0.02
        adroit_env.model.body_pos[adroit_env.target_body_id] = board_pos
        adroit_env.set_state(qpos, qvel)

        saved_state = adroit_env.get_env_state()
        saved_observation = adroit_env._get_obs()
        saved_target_pos = adroit_env.data.site_xpos[
            adroit_env.target_obj_site_id
        ].copy()

        assert "target_pos" in saved_state

        adroit_env.reset(seed=456)
        adroit_env.set_env_state(saved_state)

        np.testing.assert_allclose(adroit_env.data.qpos, saved_state["qpos"])
        np.testing.assert_allclose(adroit_env.data.qvel, saved_state["qvel"])
        np.testing.assert_allclose(
            adroit_env.model.body_pos[adroit_env.target_body_id],
            saved_state["board_pos"],
        )
        np.testing.assert_allclose(
            adroit_env.data.site_xpos[adroit_env.target_obj_site_id], saved_target_pos
        )
        np.testing.assert_allclose(adroit_env._get_obs(), saved_observation)

        documented_state = {
            key: saved_state[key] for key in ("qpos", "qvel", "board_pos")
        }

        adroit_env.reset(seed=789)
        adroit_env.set_env_state(documented_state)

        np.testing.assert_allclose(
            adroit_env.data.site_xpos[adroit_env.target_obj_site_id], saved_target_pos
        )
        np.testing.assert_allclose(adroit_env._get_obs(), saved_observation)
    finally:
        env.close()
