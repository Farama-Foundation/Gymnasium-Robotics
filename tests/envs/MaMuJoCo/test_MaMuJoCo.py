from __future__ import annotations

import collections
import os

import gymnasium
import pytest
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.utils.env_match import check_environments_match
from pettingzoo.test import parallel_api_test

import gymnasium_robotics.envs.multiagent_mujoco.many_segment_swimmer as many_segment_swimmer
from gymnasium_robotics import mamujoco_v1

scenario_conf = collections.namedtuple("scenario_conf", "scenario, conf, kwargs")

pre_defined_factorizations = [
    scenario_conf("InvertedPendulum", None, {}),  # For Debugging
    scenario_conf("Ant", None, {}),
    scenario_conf("Ant", "2x4", {}),
    scenario_conf("Ant", "2x4d", {}),
    scenario_conf("Ant", "4x2", {}),
    scenario_conf("Ant", "2x4", {}),
    scenario_conf("Ant", "2x4d", {}),
    scenario_conf(
        "Ant",
        "2x4",
        {
            "local_categories": [["qpos", "qvel"], ["qpos"], ["qpos"]],
            "include_cfrc_ext_in_observation": False,
        },
    ),
    scenario_conf(
        "Ant",
        "2x4d",
        {
            "local_categories": [["qpos", "qvel"], ["qpos"], ["qpos"]],
            "include_cfrc_ext_in_observation": False,
        },
    ),
    scenario_conf(
        "Ant",
        "4x2",
        {
            "local_categories": [["qpos", "qvel"], ["qpos"], ["qpos"]],
            "include_cfrc_ext_in_observation": False,
        },
    ),
    scenario_conf("HalfCheetah", "2x3", {}),
    scenario_conf("HalfCheetah", "6x1", {}),
    scenario_conf("HalfCheetah", None, {}),
    scenario_conf("Hopper", "3x1", {}),
    scenario_conf("Hopper", None, {}),
    scenario_conf("Humanoid", "9|8", {}),
    scenario_conf(
        "Humanoid",
        "9|8",
        {
            "local_categories": [["qpos", "qvel"], ["qpos"], ["qpos"]],
            "include_cinert_in_observation": False,
            "include_cvel_in_observation": False,
            "include_qfrc_actuator_in_observation": False,
            "include_cfrc_ext_in_observation": False,
        },
    ),
    scenario_conf("Humanoid", None, {}),
    scenario_conf("HumanoidStandup", "9|8", {}),
    scenario_conf(
        "HumanoidStandup",
        "9|8",
        {
            "local_categories": [["qpos", "qvel"], ["qpos"], ["qpos"]],
            "include_cinert_in_observation": False,
            "include_cvel_in_observation": False,
            "include_qfrc_actuator_in_observation": False,
            "include_cfrc_ext_in_observation": False,
        },
    ),
    scenario_conf("HumanoidStandup", None, {}),
    scenario_conf("Reacher", "2x1", {}),
    scenario_conf("Reacher", None, {}),
    scenario_conf("Swimmer", "2x1", {}),
    scenario_conf("Swimmer", None, {}),
    scenario_conf("Pusher", "3p", {}),
    scenario_conf("Pusher", None, {}),
    scenario_conf("Walker2d", "2x3", {}),
    scenario_conf("Walker2d", None, {}),
    scenario_conf("ManySegmentSwimmer", "10x2", {}),
    scenario_conf("ManySegmentSwimmer", "5x4", {}),
    scenario_conf("ManySegmentSwimmer", "6x1", {}),
    scenario_conf("ManySegmentSwimmer", "1x2", {}),
    scenario_conf("ManySegmentAnt", "2x3", {}),
    scenario_conf("ManySegmentAnt", "3x1", {}),
    scenario_conf("CoupledHalfCheetah", "1p1", {}),
    scenario_conf("CoupledHalfCheetah", None, {}),
]

observation_depths = [None, 0, 1, 2]


@pytest.mark.parametrize("observation_depth", observation_depths)
@pytest.mark.parametrize("task", pre_defined_factorizations)
def test_general(observation_depth, task) -> None:
    """Asserts that the environments are compliant with `pettingzoo.utils.env.ParallelEnv` API."""
    parallel_api_test(
        # MultiAgentMujocoEnv(task.scenario, task.conf, agent_obsk=observation_depth),
        mamujoco_v1.parallel_env(
            task.scenario, task.conf, agent_obsk=observation_depth, **task.kwargs
        ),
        num_cycles=1_000_000,
    )


@pytest.mark.parametrize("observation_depth", observation_depths)
@pytest.mark.parametrize("task", pre_defined_factorizations)
def test_action_and_observation_mapping(observation_depth, task):
    """Assert that converting local <-> global <-> local observations/actions results in the same observation/actions."""
    test_env = mamujoco_v1.parallel_env(
        task.scenario, task.conf, agent_obsk=observation_depth, **task.kwargs
    )

    # assert action mapping
    global_action = test_env.single_agent_env.action_space.sample()
    assert (
        global_action
        == test_env.map_local_actions_to_global_action(
            test_env.map_global_action_to_local_actions(global_action)
        )
    ).all()

    if (
        task.scenario in ["Reacher", "Pusher", "CoupledHalfCheetah"]
        and task.conf is not None
    ):
        return  # observation mapping not implemented on those environments

    # assert observation mapping
    test_env.reset()
    global_observations = test_env.state()
    local_observations = test_env.unwrapped._get_obs()
    test_env.reset()
    data_equivalence(
        test_env.map_global_state_to_local_observations(global_observations),
        local_observations,
    )

    if (
        task.scenario in ["ManySegmentSwimmer", "ManySegmentAnt"]
        and task.conf is not None
    ):
        return  # mapping local to global observation is not supported on these environments since the local observation do not observe the full environment

    data_equivalence(
        test_env.map_local_observations_to_global_state(local_observations),
        global_observations,
    )

    # sanity check making sure the observation factorizations are sane
    for agent_obs_factor in test_env.observation_factorization.values():
        len(agent_obs_factor) != len(
            set(agent_obs_factor)
        ), "an agent observes the same state value multiple times"


# The black formatter was disabled because it results in `k_dicts_tasks` being an unreadable mess
# fmt: off
pre_computed_k_dict = collections.namedtuple("pre_computed_k_dict", "scenario, conf, list_k_dicts")
k_dicts_tasks = [
    pre_computed_k_dict("Ant", "2x4", ["[{0: [ankle1, ankle2, hip1, hip2]}, {0: [ankle3, ankle4, hip3, hip4]}]", "[{0: [ankle1, ankle2, hip1, hip2], 1: [hip3, hip4]}, {0: [ankle3, ankle4, hip3, hip4], 1: [hip1, hip2]}]", "[{0: [ankle1, ankle2, hip1, hip2], 1: [hip3, hip4], 2: [ankle3, ankle4]}, {0: [ankle3, ankle4, hip3, hip4], 1: [hip1, hip2], 2: [ankle1, ankle2]}]"]),
    pre_computed_k_dict("Ant", "2x4d", ["[{0: [ankle1, ankle4, hip1, hip4]}, {0: [ankle2, ankle3, hip2, hip3]}]", "[{0: [ankle1, ankle4, hip1, hip4], 1: [hip2, hip3]}, {0: [ankle2, ankle3, hip2, hip3], 1: [hip1, hip4]}]", "[{0: [ankle1, ankle4, hip1, hip4], 1: [hip2, hip3], 2: [ankle2, ankle3]}, {0: [ankle2, ankle3, hip2, hip3], 1: [hip1, hip4], 2: [ankle1, ankle4]}]"]),
    pre_computed_k_dict("Ant", "4x2", ["[{0: [ankle1, hip1]}, {0: [ankle2, hip2]}, {0: [ankle3, hip3]}, {0: [ankle4, hip4]}]", "[{0: [ankle1, hip1], 1: [hip2, hip3, hip4]}, {0: [ankle2, hip2], 1: [hip1, hip3, hip4]}, {0: [ankle3, hip3], 1: [hip1, hip2, hip4]}, {0: [ankle4, hip4], 1: [hip1, hip2, hip3]}]", "[{0: [ankle1, hip1], 1: [hip2, hip3, hip4], 2: [ankle2, ankle3, ankle4]}, {0: [ankle2, hip2], 1: [hip1, hip3, hip4], 2: [ankle1, ankle3, ankle4]}, {0: [ankle3, hip3], 1: [hip1, hip2, hip4], 2: [ankle1, ankle2, ankle4]}, {0: [ankle4, hip4], 1: [hip1, hip2, hip3], 2: [ankle1, ankle2, ankle3]}]"]),  # noqa:E501
    pre_computed_k_dict("HalfCheetah", "2x3", ["[{0: [bfoot, bshin, bthigh]}, {0: [ffoot, fshin, fthigh]}]", "[{0: [bfoot, bshin, bthigh], 1: [fthigh]}, {0: [ffoot, fshin, fthigh], 1: [bthigh]}]", "[{0: [bfoot, bshin, bthigh], 1: [fthigh], 2: [fshin]}, {0: [ffoot, fshin, fthigh], 1: [bthigh], 2: [bshin]}]"]),
    pre_computed_k_dict("HalfCheetah", "6x1", ["[{0: [bthigh]}, {0: [bshin]}, {0: [bfoot]}, {0: [fthigh]}, {0: [fshin]}, {0: [ffoot]}]", "[{0: [bthigh], 1: [bshin, fthigh]}, {0: [bshin], 1: [bfoot, bthigh]}, {0: [bfoot], 1: [bshin]}, {0: [fthigh], 1: [bthigh, fshin]}, {0: [fshin], 1: [ffoot, fthigh]}, {0: [ffoot], 1: [fshin]}]", "[{0: [bthigh], 1: [bshin, fthigh], 2: [bfoot, fshin]}, {0: [bshin], 1: [bfoot, bthigh], 2: [fthigh]}, {0: [bfoot], 1: [bshin], 2: [bthigh]}, {0: [fthigh], 1: [bthigh, fshin], 2: [bshin, ffoot]}, {0: [fshin], 1: [ffoot, fthigh], 2: [bthigh]}, {0: [ffoot], 1: [fshin], 2: [fthigh]}]"]),  # noqa: E501
    pre_computed_k_dict("Hopper", "3x1", ["[{0: [thigh_joint]}, {0: [leg_joint]}, {0: [foot_joint]}]", "[{0: [thigh_joint], 1: [leg_joint]}, {0: [leg_joint], 1: [foot_joint, thigh_joint]}, {0: [foot_joint], 1: [leg_joint]}]", "[{0: [thigh_joint], 1: [leg_joint], 2: [foot_joint]}, {0: [leg_joint], 1: [foot_joint, thigh_joint], 2: []}, {0: [foot_joint], 1: [leg_joint], 2: [thigh_joint]}]"]),
    pre_computed_k_dict("Humanoid", "9|8", ["[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee]}]", "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2], 1: [left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee], 1: [abdomen_x, abdomen_y, abdomen_z]}]", "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2], 1: [left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z], 2: [left_knee, right_knee]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee], 1: [abdomen_x, abdomen_y, abdomen_z], 2: [left_shoulder1, left_shoulder2, right_shoulder1, right_shoulder2]}]"]),  # noqa: E501
    pre_computed_k_dict("HumanoidStandup", "9|8", ["[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee]}]", "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2], 1: [left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee], 1: [abdomen_x, abdomen_y, abdomen_z]}]", "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2], 1: [left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z], 2: [left_knee, right_knee]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee], 1: [abdomen_x, abdomen_y, abdomen_z], 2: [left_shoulder1, left_shoulder2, right_shoulder1, right_shoulder2]}]"]),  # noqa: E501
    pre_computed_k_dict("Reacher", "2x1", ["[{0: [joint0]}, {0: [joint1]}]", "[{0: [joint0], 1: [joint1]}, {0: [joint1], 1: [joint0]}]", "[{0: [joint0], 1: [joint1], 2: []}, {0: [joint1], 1: [joint0], 2: []}]"]),
    pre_computed_k_dict("Swimmer", "2x1", ["[{0: [rot2]}, {0: [rot3]}]", "[{0: [rot2], 1: [rot3]}, {0: [rot3], 1: [rot2]}]", "[{0: [rot2], 1: [rot3], 2: []}, {0: [rot3], 1: [rot2], 2: []}]"]),
    pre_computed_k_dict("Pusher", "3p", ["[{0: [r_upper_arm_roll_joint, r_wrist_roll_joint, r_wrist_roll_joint]}, {0: [r_elbow_flex_joint]}, {0: [r_forearm_roll_joint, r_wrist_flex_joint, r_wrist_roll_joint]}]", "[{0: [r_upper_arm_roll_joint, r_wrist_roll_joint, r_wrist_roll_joint], 1: [r_elbow_flex_joint]}, {0: [r_elbow_flex_joint], 1: [r_forearm_roll_joint, r_upper_arm_roll_joint]}, {0: [r_forearm_roll_joint, r_wrist_flex_joint, r_wrist_roll_joint], 1: [r_elbow_flex_joint]}]", "[{0: [r_upper_arm_roll_joint, r_wrist_roll_joint, r_wrist_roll_joint], 1: [r_elbow_flex_joint], 2: [r_forearm_roll_joint]}, {0: [r_elbow_flex_joint], 1: [r_forearm_roll_joint, r_upper_arm_roll_joint], 2: [r_wrist_flex_joint, r_wrist_roll_joint]}, {0: [r_forearm_roll_joint, r_wrist_flex_joint, r_wrist_roll_joint], 1: [r_elbow_flex_joint], 2: [r_upper_arm_roll_joint]}]"]),  # noqa: E501
    pre_computed_k_dict("Walker2d", "2x3", ["[{0: [foot_joint, leg_joint, thigh_joint]}, {0: [foot_left_joint, leg_left_joint, thigh_left_joint]}]", "[{0: [foot_joint, leg_joint, thigh_joint], 1: [thigh_left_joint]}, {0: [foot_left_joint, leg_left_joint, thigh_left_joint], 1: [thigh_joint]}]", "[{0: [foot_joint, leg_joint, thigh_joint], 1: [thigh_left_joint], 2: [leg_left_joint]}, {0: [foot_left_joint, leg_left_joint, thigh_left_joint], 1: [thigh_joint], 2: [leg_joint]}]"]),  # noqa: E501
    pre_computed_k_dict("ManySegmentSwimmer", "10x2", ["[{0: [rot0, rot1]}, {0: [rot2, rot3]}, {0: [rot4, rot5]}, {0: [rot6, rot7]}, {0: [rot8, rot9]}, {0: [rot10, rot11]}, {0: [rot12, rot13]}, {0: [rot14, rot15]}, {0: [rot16, rot17]}, {0: [rot18, rot19]}]", "[{0: [rot0, rot1], 1: [rot2]}, {0: [rot2, rot3], 1: [rot1, rot4]}, {0: [rot4, rot5], 1: [rot3, rot6]}, {0: [rot6, rot7], 1: [rot5, rot8]}, {0: [rot8, rot9], 1: [rot10, rot7]}, {0: [rot10, rot11], 1: [rot12, rot9]}, {0: [rot12, rot13], 1: [rot11, rot14]}, {0: [rot14, rot15], 1: [rot13, rot16]}, {0: [rot16, rot17], 1: [rot15, rot18]}, {0: [rot18, rot19], 1: [rot17]}]", "[{0: [rot0, rot1], 1: [rot2], 2: [rot3]}, {0: [rot2, rot3], 1: [rot1, rot4], 2: [rot0, rot5]}, {0: [rot4, rot5], 1: [rot3, rot6], 2: [rot2, rot7]}, {0: [rot6, rot7], 1: [rot5, rot8], 2: [rot4, rot9]}, {0: [rot8, rot9], 1: [rot10, rot7], 2: [rot11, rot6]}, {0: [rot10, rot11], 1: [rot12, rot9], 2: [rot13, rot8]}, {0: [rot12, rot13], 1: [rot11, rot14], 2: [rot10, rot15]}, {0: [rot14, rot15], 1: [rot13, rot16], 2: [rot12, rot17]}, {0: [rot16, rot17], 1: [rot15, rot18], 2: [rot14, rot19]}, {0: [rot18, rot19], 1: [rot17], 2: [rot16]}]"]),  # noqa: E501
    pre_computed_k_dict("ManySegmentSwimmer", "6x1", ["[{0: [rot0]}, {0: [rot1]}, {0: [rot2]}, {0: [rot3]}, {0: [rot4]}, {0: [rot5]}]", "[{0: [rot0], 1: [rot1]}, {0: [rot1], 1: [rot0, rot2]}, {0: [rot2], 1: [rot1, rot3]}, {0: [rot3], 1: [rot2, rot4]}, {0: [rot4], 1: [rot3, rot5]}, {0: [rot5], 1: [rot4]}]", "[{0: [rot0], 1: [rot1], 2: [rot2]}, {0: [rot1], 1: [rot0, rot2], 2: [rot3]}, {0: [rot2], 1: [rot1, rot3], 2: [rot0, rot4]}, {0: [rot3], 1: [rot2, rot4], 2: [rot1, rot5]}, {0: [rot4], 1: [rot3, rot5], 2: [rot2]}, {0: [rot5], 1: [rot4], 2: [rot3]}]"]),  # noqa: E501
    pre_computed_k_dict("ManySegmentAnt", "2x3", ["[{0: [ankle1_0, ankle1_1, ankle1_2, ankle2_0, ankle2_1, ankle2_2, hip1_0, hip1_1, hip1_2, hip2_0, hip2_1, hip2_2]}, {0: [ankle1_3, ankle1_4, ankle1_5, ankle2_3, ankle2_4, ankle2_5, hip1_3, hip1_4, hip1_5, hip2_3, hip2_4, hip2_5]}]", "[{0: [ankle1_0, ankle1_1, ankle1_2, ankle2_0, ankle2_1, ankle2_2, hip1_0, hip1_1, hip1_2, hip2_0, hip2_1, hip2_2], 1: [hip1_0, hip1_1, hip2_0, hip2_1]}, {0: [ankle1_3, ankle1_4, ankle1_5, ankle2_3, ankle2_4, ankle2_5, hip1_3, hip1_4, hip1_5, hip2_3, hip2_4, hip2_5], 1: [hip1_2, hip1_3, hip1_4, hip2_2, hip2_3, hip2_4]}]", "[{0: [ankle1_0, ankle1_1, ankle1_2, ankle2_0, ankle2_1, ankle2_2, hip1_0, hip1_1, hip1_2, hip2_0, hip2_1, hip2_2], 1: [hip1_0, hip1_1, hip2_0, hip2_1], 2: []}, {0: [ankle1_3, ankle1_4, ankle1_5, ankle2_3, ankle2_4, ankle2_5, hip1_3, hip1_4, hip1_5, hip2_3, hip2_4, hip2_5], 1: [hip1_2, hip1_3, hip1_4, hip2_2, hip2_3, hip2_4], 2: []}]"]),  # noqa: E501
    pre_computed_k_dict("ManySegmentAnt", "3x1", ["[{0: [ankle1_0, ankle2_0, hip1_0, hip2_0]}, {0: [ankle1_1, ankle2_1, hip1_1, hip2_1]}, {0: [ankle1_2, ankle2_2, hip1_2, hip2_2]}]", "[{0: [ankle1_0, ankle2_0, hip1_0, hip2_0], 1: []}, {0: [ankle1_1, ankle2_1, hip1_1, hip2_1], 1: [hip1_0, hip2_0]}, {0: [ankle1_2, ankle2_2, hip1_2, hip2_2], 1: [hip1_1, hip2_1]}]", "[{0: [ankle1_0, ankle2_0, hip1_0, hip2_0], 1: [], 2: []}, {0: [ankle1_1, ankle2_1, hip1_1, hip2_1], 1: [hip1_0, hip2_0], 2: []}, {0: [ankle1_2, ankle2_2, hip1_2, hip2_2], 1: [hip1_1, hip2_1], 2: []}]"]),  # noqa: E501
    pre_computed_k_dict("CoupledHalfCheetah", "1p1", ["[{0: [bfoot0, bshin0, bthigh0, ffoot0, fshin0, fthigh0]}, {0: [bfoot1, bshin1, bthigh1, ffoot1, fshin1, fthigh1]}]", "[{0: [bfoot0, bshin0, bthigh0, ffoot0, fshin0, fthigh0], 1: []}, {0: [bfoot1, bshin1, bthigh1, ffoot1, fshin1, fthigh1], 1: []}]", "[{0: [bfoot0, bshin0, bthigh0, ffoot0, fshin0, fthigh0], 1: [], 2: []}, {0: [bfoot1, bshin1, bthigh1, ffoot1, fshin1, fthigh1], 1: [], 2: []}]"]),
]
# fmt: on


@pytest.mark.parametrize("task", k_dicts_tasks)
def test_k_dict(task):
    """Asserts that `obsk.get_joints_at_kdist()` generates the correct observation factorization.

    The outputs have been hand written
    If this test fails it means either the factorization in `obsk.get_parts_and_edges()` is wrong or that `obsk.get_joints_at_kdist()` generates wrong k_dict
    """
    for k, k_dict in enumerate(task.list_k_dicts):
        test_env = mamujoco_v1.parallel_env(
            scenario=task.scenario, agent_conf=task.conf, agent_obsk=k
        )
        assert str(test_env.k_dicts) == k_dict, str(test_env.k_dicts)


def test_swimmer_gen():
    """Assert that the many segment swimmer environment is identical to the simple environments."""
    env = gymnasium.make("Swimmer-v5")

    asset_path = "/tmp/swimmer_2seg.xml"
    many_segment_swimmer.gen_asset(n_segs=2, asset_path=asset_path)
    c_env = gymnasium.make("Swimmer-v5", xml_file=asset_path)
    os.remove(asset_path)

    check_environments_match(env, c_env, num_steps=2000)
