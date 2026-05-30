import gymnasium as gym
import numpy as np

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


def test_set_env_state_preserves_relocated_object_position():
    env = gym.make("AdroitHandRelocate-v2", disable_env_checker=True)
    adroit_env = env.unwrapped

    try:
        adroit_env.reset(seed=123)

        qpos = adroit_env.data.qpos.copy()
        qvel = adroit_env.data.qvel.copy()
        object_translation = np.array([0.03, -0.02, 0.1])
        object_body_pos = np.array([0.12, -0.08, 0.04])
        target_pos = np.array([0.04, 0.05, 0.2])

        qpos[adroit_env.obj_translation_qpos_indices] = object_translation
        adroit_env.model.body_pos[adroit_env.obj_body_id] = object_body_pos
        adroit_env.model.site_pos[adroit_env.target_obj_site_id] = target_pos
        adroit_env.set_state(qpos, qvel)

        saved_state = adroit_env.get_env_state()
        saved_observation = adroit_env._get_obs()
        expected_obj_pos = object_body_pos + object_translation

        np.testing.assert_allclose(saved_state["obj_pos"], expected_obj_pos)

        adroit_env.reset(seed=456)
        adroit_env.set_env_state(saved_state)

        np.testing.assert_allclose(
            adroit_env.data.xpos[adroit_env.obj_body_id], expected_obj_pos
        )
        np.testing.assert_allclose(adroit_env.data.qpos, saved_state["qpos"])
        np.testing.assert_allclose(adroit_env.data.qvel, saved_state["qvel"])
        np.testing.assert_allclose(adroit_env._get_obs(), saved_observation)

        documented_state = {
            key: saved_state[key] for key in ("qpos", "qvel", "obj_pos", "target_pos")
        }

        adroit_env.reset(seed=789)
        adroit_env.set_env_state(documented_state)

        np.testing.assert_allclose(
            adroit_env.data.xpos[adroit_env.obj_body_id], expected_obj_pos
        )
        np.testing.assert_allclose(adroit_env._get_obs(), saved_observation)
    finally:
        env.close()
