"""Check ManySegmentAnt node indices against its generated MuJoCo model."""

import numpy as np
import pytest

from gymnasium_robotics import mamujoco_v1


@pytest.fixture(params=["1x1", "3x1", "2x3"])
def ant_env(request):
    """Create factorizations with one and multiple segments per agent."""
    env = mamujoco_v1.parallel_env(
        "ManySegmentAnt",
        request.param,
        agent_obsk=0,
        local_categories=[["qpos", "qvel"]],
        global_categories=(),
    )
    yield env
    env.close()


@pytest.mark.parametrize("coordinate", ["qpos", "qvel", "act"])
def test_node_indices_match_model(ant_env, coordinate):
    """Each node must read and actuate the joint named by its label."""
    model = ant_env.single_agent_env.unwrapped.model
    size = {"qpos": model.nq, "qvel": model.nv, "act": model.nu}[coordinate]
    for partition in ant_env.agent_action_partitions:
        for node in partition:
            joint = model.joint(node.label)
            expected = {
                "qpos": joint.qposadr[0],
                "qvel": joint.dofadr[0],
                "act": np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)[0],
            }[coordinate]
            actual = getattr(node, f"{coordinate}_ids") % size
            assert actual == expected, (node.label, coordinate, actual, expected)


def test_local_observations_use_joint_coordinates(ant_env):
    """Compare observations to named joint addresses, not the node indices."""
    ant_env.reset(seed=123)
    base_env = ant_env.single_agent_env.unwrapped
    base_env.data.qpos[:] = np.arange(base_env.model.nq) * 0.01
    base_env.data.qvel[:] = np.arange(base_env.model.nv) * -0.01
    observations = ant_env._get_obs()
    for agent, k_dict in zip(ant_env.possible_agents, ant_env.k_dicts):
        expected = []
        for node in k_dict[0]:
            joint = base_env.model.joint(node.label)
            expected.extend(
                [
                    base_env.data.qpos[joint.qposadr[0]],
                    base_env.data.qvel[joint.dofadr[0]],
                ]
            )
        np.testing.assert_allclose(observations[agent], expected)
