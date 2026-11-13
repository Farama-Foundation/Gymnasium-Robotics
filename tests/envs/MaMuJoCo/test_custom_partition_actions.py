import numpy as np

from gymnasium_robotics import mamujoco_v1


def test_custom_partition_routes_actions_without_conf():
    parts, edges, globals_ = mamujoco_v1.get_parts_and_edges("Ant", "2x4")
    nodes = [node for part in parts for node in part][::-1]
    partitions = [tuple(nodes[:1]), tuple(nodes[1:4]), tuple(nodes[4:])]
    env = mamujoco_v1.parallel_env(
        "Ant",
        None,
        agent_obsk=1,
        agent_factorization={
            "partition": partitions,
            "edges": edges,
            "globals": globals_,
        },
    )
    try:
        observations, _ = env.reset(seed=7)
        assert len(observations) == 3
        expected = np.linspace(-0.7, 0.7, 8, dtype=np.float32)
        actions = {
            agent: expected[[node.act_ids for node in part]]
            for agent, part in zip(env.possible_agents, partitions)
        }
        for agent, action in actions.items():
            assert env.action_space(agent).contains(action)
        observations, rewards, terminated, truncated, infos = env.step(actions)
        assert set(observations) == set(actions)
        np.testing.assert_allclose(env.single_agent_env.unwrapped.data.ctrl, expected)
    finally:
        env.close()
