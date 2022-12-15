from __future__ import annotations

import collections

import numpy
import pytest
from multiagent_mujoco.mujoco_multi import MultiAgentMujocoEnv
from pettingzoo.test import parallel_api_test

scenario_conf = collections.namedtuple("scenario_conf", "scenario, conf")

pre_defined_factorizations = [
    scenario_conf("InvertedPendulum", None),  # For Debugging
    scenario_conf("Ant", None),
    scenario_conf("Ant", "2x4"),
    scenario_conf("Ant", "2x4d"),
    scenario_conf("Ant", "4x2"),
    scenario_conf("HalfCheetah", "2x3"),
    scenario_conf("HalfCheetah", "6x1"),
    scenario_conf("HalfCheetah", None),
    scenario_conf("Hopper", "3x1"),
    scenario_conf("Hopper", None),
    scenario_conf("Humanoid", "9|8"),
    scenario_conf("Humanoid", None),
    scenario_conf("HumanoidStandup", "9|8"),
    scenario_conf("HumanoidStandup", None),
    scenario_conf("Reacher", "2x1"),
    scenario_conf("Reacher", None),
    scenario_conf("Swimmer", "2x1"),
    scenario_conf("Swimmer", None),
    scenario_conf("Walker2d", "2x3"),
    scenario_conf("Walker2d", None),
]

sample_configurations = [
    scenario_conf("manyagent_swimmer", "10x2"),
    scenario_conf("manyagent_swimmer", "6x1"),
    scenario_conf("manyagent_ant", "2x3"),
    scenario_conf("manyagent_ant", "3x1"),
    scenario_conf("coupled_half_cheetah", "1p1"),
    scenario_conf("coupled_half_cheetah", None),
]

observation_depths = [None, 0, 1, 2]


def assert_dict_numpy_are_equal(
    dict_a: dict[any, numpy.array], dict_b: dict[any, numpy.array]
) -> None:
    assert dict_a.keys() == dict_b.keys()
    for key in dict_a.keys():
        assert (dict_a[key] == dict_b[key]).all()


@pytest.mark.parametrize("observation_depth", observation_depths)
@pytest.mark.parametrize("task", pre_defined_factorizations + sample_configurations)
def test_general(observation_depth, task):
    parallel_api_test(
        MultiAgentMujocoEnv(task.scenario, task.conf, agent_obsk=observation_depth),
        num_cycles=1_000_000,
    )


@pytest.mark.parametrize("observation_depth", observation_depths)
@pytest.mark.parametrize("task", pre_defined_factorizations)
def test_action_and_observation_mapping(observation_depth, task):
    """
    assert that converting local <-> global <-> local obervations/actions results in the same observation/actions
    """
    test_env = MultiAgentMujocoEnv(
        task.scenario, task.conf, agent_obsk=observation_depth
    )
    # assert action mapping
    global_action = test_env.gym_env.action_space.sample()
    assert (
        global_action
        == test_env.map_local_actions_to_global_action(
            test_env.map_global_action_to_local_actions(global_action)
        )
    ).all()

    if task == scenario_conf("Reacher", "2x1"):
        return  # observation mapping not implemented on 'Reacher' Environment

    # assert observation mapping
    test_env.reset()
    global_observations = test_env.state()
    local_observations = test_env.unwrapped._get_obs()
    test_env.reset()
    assert_dict_numpy_are_equal(
        test_env.map_global_state_to_local_observations(global_observations),
        local_observations,
    )


@pytest.mark.parametrize("observation_depth", observation_depths)
@pytest.mark.parametrize("task", sample_configurations)
def test_action_mapping(observation_depth, task):
    # observation mapping not implemented non-Gymansium mujoco environments
    """
    assert that converting local <-> global <-> local actions results in the same actions
    """
    test_env = MultiAgentMujocoEnv(
        task.scenario, task.conf, agent_obsk=observation_depth
    )
    global_action = test_env.gym_env.action_space.sample()
    assert (
        global_action
        == test_env.map_local_actions_to_global_action(
            test_env.map_global_action_to_local_actions(global_action)
        )
    ).all()


def test_k_dict():
    """
    asserts that obsk.get_joints_at_kdist() generates the correct observation factorization
    The outputs have been hand written
    """
    scenario = "Ant"
    agent_conf = "2x4"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, ankle2, hip1, hip2]}, {0: [ankle3, ankle4, hip3, hip4]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, ankle2, hip1, hip2], 1: [hip3, hip4]}, {0: [ankle3, ankle4, hip3, hip4], 1: [hip1, hip2]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, ankle2, hip1, hip2], 1: [hip3, hip4], 2: [ankle3, ankle4]}, {0: [ankle3, ankle4, hip3, hip4], 1: [hip1, hip2], 2: [ankle1, ankle2]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "Ant"
    agent_conf = "2x4d"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, ankle3, hip1, hip3]}, {0: [ankle2, ankle4, hip2, hip4]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, ankle3, hip1, hip3], 1: [hip2, hip4]}, {0: [ankle2, ankle4, hip2, hip4], 1: [hip1, hip3]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, ankle3, hip1, hip3], 1: [hip2, hip4], 2: [ankle2, ankle4]}, {0: [ankle2, ankle4, hip2, hip4], 1: [hip1, hip3], 2: [ankle1, ankle3]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "Ant"
    agent_conf = "4x2"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, hip1]}, {0: [ankle2, hip2]}, {0: [ankle3, hip3]}, {0: [ankle4, hip4]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, hip1], 1: [hip2, hip3, hip4]}, {0: [ankle2, hip2], 1: [hip1, hip3, hip4]}, {0: [ankle3, hip3], 1: [hip1, hip2, hip4]}, {0: [ankle4, hip4], 1: [hip1, hip2, hip3]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1, hip1], 1: [hip2, hip3, hip4], 2: [ankle2, ankle3, ankle4]}, {0: [ankle2, hip2], 1: [hip1, hip3, hip4], 2: [ankle1, ankle3, ankle4]}, {0: [ankle3, hip3], 1: [hip1, hip2, hip4], 2: [ankle1, ankle2, ankle4]}, {0: [ankle4, hip4], 1: [hip1, hip2, hip3], 2: [ankle1, ankle2, ankle3]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "HalfCheetah"
    agent_conf = "2x3"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot, bshin, bthigh]}, {0: [ffoot, fshin, fthigh]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot, bshin, bthigh], 1: [fthigh]}, {0: [ffoot, fshin, fthigh], 1: [bthigh]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot, bshin, bthigh], 1: [fthigh], 2: [fshin]}, {0: [ffoot, fshin, fthigh], 1: [bthigh], 2: [bshin]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "HalfCheetah"
    agent_conf = "6x1"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot]}, {0: [bshin]}, {0: [bthigh]}, {0: [ffoot]}, {0: [fshin]}, {0: [fthigh]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot], 1: [bshin]}, {0: [bshin], 1: [bfoot, bthigh]}, {0: [bthigh], 1: [bshin, fthigh]}, {0: [ffoot], 1: [fshin]}, {0: [fshin], 1: [ffoot, fthigh]}, {0: [fthigh], 1: [bthigh, fshin]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot], 1: [bshin], 2: [bthigh]}, {0: [bshin], 1: [bfoot, bthigh], 2: [fthigh]}, {0: [bthigh], 1: [bshin, fthigh], 2: [bfoot, fshin]}, {0: [ffoot], 1: [fshin], 2: [fthigh]}, {0: [fshin], 1: [ffoot, fthigh], 2: [bthigh]}, {0: [fthigh], 1: [bthigh, fshin], 2: [bshin, ffoot]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "Hopper"
    agent_conf = "3x1"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [thigh_joint]}, {0: [leg_joint]}, {0: [foot_joint]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [thigh_joint], 1: [leg_joint]}, {0: [leg_joint], 1: [foot_joint, thigh_joint]}, {0: [foot_joint], 1: [leg_joint]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [thigh_joint], 1: [leg_joint], 2: [foot_joint]}, {0: [leg_joint], 1: [foot_joint, thigh_joint], 2: []}, {0: [foot_joint], 1: [leg_joint], 2: [thigh_joint]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "Humanoid"
    agent_conf = "9|8"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2], 1: [left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee], 1: [abdomen_x, abdomen_y, abdomen_z]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert str(test_env.k_dicts) == (
        "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2], 1: [left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z], 2: [left_knee, right_knee]}, "
        "{0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee], 1: [abdomen_x, abdomen_y, abdomen_z], 2: [left_shoulder1, left_shoulder2, right_shoulder1, right_shoulder2]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "HumanoidStandup"
    agent_conf = "9|8"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2], 1: [left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z]}, {0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee], 1: [abdomen_x, abdomen_y, abdomen_z]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert str(test_env.k_dicts) == (
        "[{0: [abdomen_x, abdomen_y, abdomen_z, left_elbow, left_shoulder1, left_shoulder2, right_elbow, right_shoulder1, right_shoulder2], 1: [left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z], 2: [left_knee, right_knee]}, "
        "{0: [left_hip_x, left_hip_y, left_hip_z, left_knee, right_hip_x, right_hip_y, right_hip_z, right_knee], 1: [abdomen_x, abdomen_y, abdomen_z], 2: [left_shoulder1, left_shoulder2, right_shoulder1, right_shoulder2]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "Reacher"
    agent_conf = "2x1"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts) == "[{0: [joint0]}, {0: [joint1]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [joint0], 1: [joint1]}, {0: [joint1], 1: [joint0]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [joint0], 1: [joint1], 2: []}, {0: [joint1], 1: [joint0], 2: []}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "Swimmer"
    agent_conf = "2x1"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts) == "[{0: [rot2]}, {0: [rot3]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts) == "[{0: [rot2], 1: [rot3]}, {0: [rot3], 1: [rot2]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [rot2], 1: [rot3], 2: []}, {0: [rot3], 1: [rot2], 2: []}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "Walker2d"
    agent_conf = "2x3"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [foot_joint, leg_joint, thigh_joint]}, {0: [foot_left_joint, leg_left_joint, thigh_left_joint]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [foot_joint, leg_joint, thigh_joint], 1: [thigh_left_joint]}, {0: [foot_left_joint, leg_left_joint, thigh_left_joint], 1: [thigh_joint]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [foot_joint, leg_joint, thigh_joint], 1: [thigh_left_joint], 2: [leg_left_joint]}, {0: [foot_left_joint, leg_left_joint, thigh_left_joint], 1: [thigh_joint], 2: [leg_joint]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "manyagent_swimmer"
    agent_conf = "10x2"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [rot0, rot1]}, {0: [rot2, rot3]}, {0: [rot4, rot5]}, {0: [rot6, rot7]}, {0: [rot8, rot9]}, {0: [rot10, rot11]}, {0: [rot12, rot13]}, {0: [rot14, rot15]}, {0: [rot16, rot17]}, {0: [rot18, rot19]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [rot0, rot1], 1: [rot2]}, {0: [rot2, rot3], 1: [rot1, rot4]}, {0: [rot4, rot5], 1: [rot3, rot6]}, {0: [rot6, rot7], 1: [rot5, rot8]}, {0: [rot8, rot9], 1: [rot10, rot7]}, {0: [rot10, rot11], 1: [rot12, rot9]}, {0: [rot12, rot13], 1: [rot11, rot14]}, {0: [rot14, rot15], 1: [rot13, rot16]}, {0: [rot16, rot17], 1: [rot15, rot18]}, {0: [rot18, rot19], 1: [rot17]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert str(test_env.k_dicts) == (
        "[{0: [rot0, rot1], 1: [rot2], 2: [rot3]}, {0: [rot2, rot3], 1: [rot1, rot4], 2: [rot0, rot5]}, {0: [rot4, rot5], 1: [rot3, rot6], 2: [rot2, rot7]}, {0: [rot6, rot7], 1: [rot5, rot8], 2: [rot4, rot9]}, {0: [rot8, rot9], 1: [rot10, rot7], 2: [rot11, rot6]}, "
        "{0: [rot10, rot11], 1: [rot12, rot9], 2: [rot13, rot8]}, {0: [rot12, rot13], 1: [rot11, rot14], 2: [rot10, rot15]}, {0: [rot14, rot15], 1: [rot13, rot16], 2: [rot12, rot17]}, {0: [rot16, rot17], 1: [rot15, rot18], 2: [rot14, rot19]}, {0: [rot18, rot19], 1: [rot17], 2: [rot16]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    agent_conf = "6x1"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [rot0]}, {0: [rot1]}, {0: [rot2]}, {0: [rot3]}, {0: [rot4]}, {0: [rot5]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [rot0], 1: [rot1]}, {0: [rot1], 1: [rot0, rot2]}, {0: [rot2], 1: [rot1, rot3]}, {0: [rot3], 1: [rot2, rot4]}, {0: [rot4], 1: [rot3, rot5]}, {0: [rot5], 1: [rot4]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [rot0], 1: [rot1], 2: [rot2]}, {0: [rot1], 1: [rot0, rot2], 2: [rot3]}, {0: [rot2], 1: [rot1, rot3], 2: [rot0, rot4]}, {0: [rot3], 1: [rot2, rot4], 2: [rot1, rot5]}, {0: [rot4], 1: [rot3, rot5], 2: [rot2]}, {0: [rot5], 1: [rot4], 2: [rot3]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "manyagent_ant"
    agent_conf = "2x3"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1_0, ankle1_1, ankle1_2, ankle2_0, ankle2_1, ankle2_2, hip1_0, hip1_1, hip1_2, hip2_0, hip2_1, hip2_2]}, {0: [ankle1_3, ankle1_4, ankle1_5, ankle2_3, ankle2_4, ankle2_5, hip1_3, hip1_4, hip1_5, hip2_3, hip2_4, hip2_5]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1_0, ankle1_1, ankle1_2, ankle2_0, ankle2_1, ankle2_2, hip1_0, hip1_1, hip1_2, hip2_0, hip2_1, hip2_2], 1: [hip1_0, hip1_1, hip2_0, hip2_1]}, {0: [ankle1_3, ankle1_4, ankle1_5, ankle2_3, ankle2_4, ankle2_5, hip1_3, hip1_4, hip1_5, hip2_3, hip2_4, hip2_5], 1: [hip1_2, hip1_3, hip1_4, hip2_2, hip2_3, hip2_4]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1_0, ankle1_1, ankle1_2, ankle2_0, ankle2_1, ankle2_2, hip1_0, hip1_1, hip1_2, hip2_0, hip2_1, hip2_2], 1: [hip1_0, hip1_1, hip2_0, hip2_1], 2: []}, {0: [ankle1_3, ankle1_4, ankle1_5, ankle2_3, ankle2_4, ankle2_5, hip1_3, hip1_4, hip1_5, hip2_3, hip2_4, hip2_5], 1: [hip1_2, hip1_3, hip1_4, hip2_2, hip2_3, hip2_4], 2: []}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    agent_conf = "3x1"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1_0, ankle2_0, hip1_0, hip2_0]}, {0: [ankle1_1, ankle2_1, hip1_1, hip2_1]}, {0: [ankle1_2, ankle2_2, hip1_2, hip2_2]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1_0, ankle2_0, hip1_0, hip2_0], 1: []}, {0: [ankle1_1, ankle2_1, hip1_1, hip2_1], 1: [hip1_0, hip2_0]}, {0: [ankle1_2, ankle2_2, hip1_2, hip2_2], 1: [hip1_1, hip2_1]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [ankle1_0, ankle2_0, hip1_0, hip2_0], 1: [], 2: []}, {0: [ankle1_1, ankle2_1, hip1_1, hip2_1], 1: [hip1_0, hip2_0], 2: []}, {0: [ankle1_2, ankle2_2, hip1_2, hip2_2], 1: [hip1_1, hip2_1], 2: []}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)

    scenario = "coupled_half_cheetah"
    agent_conf = "1p1"
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=0
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot, bshin, bthigh, ffoot, fshin, fthigh]}, {0: [bfoot2, bshin2, bthigh2, ffoot2, fshin2, fthigh2]}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=1
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot, bshin, bthigh, ffoot, fshin, fthigh], 1: []}, {0: [bfoot2, bshin2, bthigh2, ffoot2, fshin2, fthigh2], 1: []}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
    test_env = MultiAgentMujocoEnv(
        scenario=scenario, agent_conf=agent_conf, agent_obsk=2
    )
    assert (
        str(test_env.k_dicts)
        == "[{0: [bfoot, bshin, bthigh, ffoot, fshin, fthigh], 1: [], 2: []}, {0: [bfoot2, bshin2, bthigh2, ffoot2, fshin2, fthigh2], 1: [], 2: []}]"
    ), "wrong k_dicts: " + str(test_env.k_dicts)
