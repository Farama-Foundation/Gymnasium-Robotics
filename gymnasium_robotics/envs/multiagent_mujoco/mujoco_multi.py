from __future__ import annotations

import gymnasium
import numpy as np
import pettingzoo
from gymnasium.wrappers.time_limit import TimeLimit
from multiagent_mujoco.coupled_half_cheetah import CoupledHalfCheetah
from multiagent_mujoco.many_segment_ant import ManySegmentAntEnv
from multiagent_mujoco.many_segment_swimmer import ManySegmentSwimmerEnv
from multiagent_mujoco.obsk import (
    _observation_structure,
    build_obs,
    get_joints_at_kdist,
    get_parts_and_edges,
)

# TODO for future revisions v1?
# color the renderer
# support other Gymnasium-Robotics MuJoCo environments

_MUJOCO_GYM_ENVIROMENTS = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "HumanoidStandup-v4",
    "Humanoid-v4",
    "Reacher-v4",
    "Swimmer-v4",
    "Pusher-v4",
    "Walker2d-v4",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
]


class MultiAgentMujocoEnv(pettingzoo.utils.env.ParallelEnv):
    """Class for multi agent factorizing mujoco environments.

    # MaMuJoCo (Multi-Agent MuJoCo)

    These environments were introduced in ["FACMAC: Factored Multi-Agent Centralised Policy Gradients"](https://arxiv.org/abs/2003.06709)

    There are 2 types of Environments, included (1) multi-agent factorizations of [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/) tasks and (2) new complex MuJoCo tasks meant to me solved with multi-agent Algorithms

    This Represents the first, easy to use Framework for research of agent factorization

    # API

    MaMuJoCo uses the [PettingZoo.ParallelAPI](https://pettingzoo.farama.org/api/parallel/), but also supports a few extra functions
    - MaMuJoCo.map_local_actions_to_global_action
    - MaMuJoCo.map_global_action_to_local_actions
    - MaMuJoCo.map_global_state_to_local_observations
    - MaMuJoCo.map_local_observation_to_global_state (NOT IMPLEMENTED)
    - obsk.get_parts_and_edges

    # Action Spaces

    For (1) the action space shape is the shape of the single agent domain divided by the number of agents

    For (2) it Depends on the configuration

    # State Spaces

    Depends on the Environment

    # Rewards

    For (1) uses the same rewards of single agent gymnasium for each agent

    For (2) used the same reward structure as the 'simpler' equivalent agents but scaled

    # Starting State

    For (1) uses the same starting state as the single agent gymnasium equivalent

    For (2) used the same starting state structure as the 'simpler' equivalent agents

    # Episode End

    For (1) uses the same termination and truncation mechanism as the single agent gymnasium (Note: all the agents terminate and truncation at the same time)

    For manyagent_swimmer
        truncates all agents at 1000 steps, and never terminates
    For manyagent_ant
        truncates all agents at 1000 steps, and never terminates based on same condition as "Ant"
    For coupled_half_cheetah
        truncates all agents at 1000 steps, and never terminates


    # Valid pre-made Configurations

    ### 2-Agent Ant

    scenario="Ant-v2"
    agent_conf="2x4"

    ### 2-Agent Ant Diag

    scenario="Ant-v2"
    agent_conf="2x4d"

    ### 4-Agent Ant

    scenario="Ant-v2"
    agent_conf="4x2"

    ### 2-Agent HalfCheetah

    scenario="HalfCheetah-v2"
    agent_conf="2x3"

    ### 6-Agent HalfCheetah

    scenario="HalfCheetah-v2"
    agent_conf="6x1"

    ### 3-Agent Hopper

    scenario="Hopper-v2"
    agent_conf="3x1"

    ### 2-Agent Humanoid

    scenario="Humanoid-v2"
    agent_conf="9|8"

    ### 2-Agent HumanoidStandup

    scenario="HumanoidStandup-v2"
    agent_conf="9|8"

    ### 2-Agent Reacher

    scenario="Reacher-v2"
    agent_conf="2x1"

    ### 2-Agent Swimmer

    scenario="Swimmer-v2"
    agent_conf="2x1"

    ### 2-Agent Walker

    scenario="Walker2d-v2"
    agent_conf="2x3"

    ### 1-Agent InvertedPendulum (for debugging algorithms)
    scenario="InvertedPendulum"
    agent_conf=None

    ### Manyagent Swimmer

    scenario="manyagent_swimmer"
    agent_conf="10x2"

    scenario="manyagent_swimmer"
    agent_conf="$Xx$Y" # where $X, $Y any positive integers e,g, "42x6", "10x2", "2x3"


    ### Manyagent Ant

    scenario="manyagent_ant"
    agent_conf="2x3"

    scenario="manyagent_ant"
    agent_conf="$Xx$Y" # where $X, $Y any positive integers e,g, "42x6", "10x2", "2x3"

    ### Coupled HalfCheetah (NEW!)

    scenario="coupled_half_cheetah"
    agent_conf="1p1"


    # How to create new agent factorizations (example 'Ant-v4', '8x1')

    In this example, we will create an agent factorization not present in MaMuJoCo the '8x1', where each agent controls a single action (first implemented by [safe-MaMuJoCo](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco))

    first we will load the graph of MaMuJoCo
    ```python
    >>> from multiagent_mujoco.obsk import get_parts_and_edges
    >>> unpartioned_nodes, edges, global_nodes = get_parts_and_edges('Ant-v4', None)
    ```
    the `unpartioned_nodes` contain the nodes of the MaMuJoCo graph
    the `edges` well, contain the edges of the graph
    and the `global_nodes` a set of observations for all agents

    To create our '8x1' partition we will need to partition the `unpartioned_nodes`

    ```python
    >>> unpartioned_nodes
    [(hip1, ankle1, hip2, ankle2, hip3, ankle3, hip4, ankle4)]
    >>> partioned_nodes = [(unpartioned_nodes[0][0],), (unpartioned_nodes[0][1],), (unpartioned_nodes[0][2],), (unpartioned_nodes[0][3],), (unpartioned_nodes[0][4],), (unpartioned_nodes[0][5],), (unpartioned_nodes[0][6],), (unpartioned_nodes[0][7],)]>>> partioned_nodes
    >>> partioned_nodes
    [(hip1,), (ankle1,), (hip2,), (ankle2,), (hip3,), (ankle3,), (hip4,), (ankle4,)]
    ```
    finally package the partitions and create our environment
    ```python
    my_agent_factorization = {"partition": partioned_nodes, "edges": edges, "globals": global_nodes}
    gym_env = MaMuJoCo('Ant', '8x1', agent_factorization=my_agent_factorization)
    ```
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "name": "MaMuJoCo",
        "is_parallelizable": True,
        # "render_fps": 0,  #depends on underlying Envrioment
        "has_manual_policy": False,
    }

    def __init__(
        self,
        scenario: str,
        agent_conf: str,
        agent_obsk: int = 1,
        agent_factorization: dict[str, any] = None,
        local_categories: list[list[str]] = None,
        global_categories: list[str] = None,
        render_mode: str = None,
    ):
        """Init.

        Args:
            scenario: The Task/Environment, valid values:
                "Ant", "HalfCheetah", "Hopper", "HumanoidStandup", "Humanoid", "Reacher", "Swimmer", "Walker2d", "InvertedPendulum", "InvertedDoublePendulum", "manyagent_swimmer", "manyagent_ant", "coupled_half_cheetah"
            agent_conf: '${Number Of Agents}x${Number Of Segments per Agent}${Optionally Additional options}', eg '1x6', '2x4', '2x4d',
                if it set to None the task becomes single agent (the agent observes the entire environment, and performs all the actions)
            agent_obsk: Number of nearest joints to observe,
                if set to 0 it only observes local state,
                if set to 1 it observes local state + 1 joint over,
                if set to 2 it observes local state + 2 joints over,
                if it set to None the task becomes single agent (the agent observes the entire environment, and performs all the actions)
                The Default value is: 1
            agent_factorization: A custom factorization of the MuJoCo environment (overwrites agent_conf),
                see DOC [how to create new agent factorizations](link).
            local_categories: The categories of local observations for each observation depth,
                The default is: Everything is observable at depth 0, but only the position items are observable for further depth levels
            global_categories: The categories of observation for global observations,
                The default is; local_categories[0]
            render_mode: see [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/),
                valid values: 'human', 'rgb_array', 'depth_array'

            Raises: NotImplementedError: When the scenario is not supported (not part of of the valid values)
        """
        scenario += "-v4"

        # load the underlying single agent Gymansium MuJoCo Environment in `self.gym_env`
        if scenario in _MUJOCO_GYM_ENVIROMENTS:
            self.single_agent_env = gymnasium.make(scenario, render_mode=render_mode)
        elif scenario in ["manyagent_ant-v4"]:
            try:
                n_segs = int(agent_conf.split("x")[0]) * int(agent_conf.split("x")[1])
            except Exception:
                raise Exception(f"UNKNOWN partitioning config: {agent_conf}")

            self.single_agent_env = TimeLimit(
                ManySegmentAntEnv(n_segs, render_mode), max_episode_steps=1000
            )
        elif scenario in ["manyagent_swimmer-v4"]:
            try:
                n_segs = int(agent_conf.split("x")[0]) * int(agent_conf.split("x")[1])
            except Exception:
                raise Exception(f"UNKNOWN partitioning config: {agent_conf}")

            self.single_agent_env = TimeLimit(
                ManySegmentSwimmerEnv(n_segs, render_mode), max_episode_steps=1000
            )
        elif scenario in ["coupled_half_cheetah-v4"]:
            self.single_agent_env = TimeLimit(
                CoupledHalfCheetah(render_mode), max_episode_steps=1000
            )
        else:
            raise NotImplementedError("Custom env not implemented!")

        if agent_conf is None:
            self.agent_obsk = None
        else:
            self.agent_obsk = agent_obsk  # if None, fully observable else k>=0 implies observe nearest k agents or joints

        # load the agent factorization
        if self.agent_obsk is not None:
            if agent_factorization is None:
                (
                    self.agent_action_partitions,
                    mujoco_edges,
                    self.mujoco_globals,
                ) = get_parts_and_edges(scenario, agent_conf)
            else:
                self.agent_action_partitions = agent_factorization["partition"]
                mujoco_edges = agent_factorization["edges"]
                self.mujoco_globals = agent_factorization["globals"]
        else:
            assert self.single_agent_env.action_space.shape is not None
            self.agent_action_partitions = [
                tuple(None for i in range(self.single_agent_env.action_space.shape[0]))
            ]
            mujoco_edges = None

        self.possible_agents = [
            str(agent_id) for agent_id in range(len(self.agent_action_partitions))
        ]
        self.agents = self.possible_agents

        # load the observation categories
        if local_categories is None:
            self.k_categories = self._generate_local_categories(scenario)
        else:
            self.k_categories = local_categories
        if global_categories is None:
            self.global_categories = self._generate_global_categories(scenario)
        else:
            self.global_categories = global_categories

        # load the observations per depth level
        self.k_dicts = [
            get_joints_at_kdist(
                self.agent_action_partitions[agent_id],
                mujoco_edges,
                k=self.agent_obsk,
            )
            for agent_id in range(self.num_agents)
        ]

        self.observation_spaces, self.action_spaces = {}, {}
        for agent_id, partition in enumerate(self.agent_action_partitions):
            self.action_spaces[self.possible_agents[agent_id]] = gymnasium.spaces.Box(
                low=self.single_agent_env.action_space.low[0],
                high=self.single_agent_env.action_space.high[0],
                shape=(len(partition),),
                dtype=np.float32,
            )
            self.observation_spaces[
                self.possible_agents[agent_id]
            ] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self._get_obs_agent(agent_id)),),
                dtype=self.single_agent_env.observation_space.dtype,
            )

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, str],
    ]:
        """Runs one timestep of the environment using the agents's actions.

        Note: if step is called after the agents have terminated/truncated the envrioment will continue to work as normal
        Args:
            actions:
                the actions of all agents
        Returns:
            see pettingzoo.utils.env.ParallelEnv.step() doc
        """
        _, reward_n, is_terminal_n, is_truncated_n, info_n = self.single_agent_env.step(
            self.map_local_actions_to_global_action(actions)
        )

        rewards, terminations, truncations, info = {}, {}, {}, {}
        observations = self._get_obs()
        for agents in self.possible_agents:
            rewards[agents] = reward_n
            terminations[agents] = is_terminal_n
            truncations[agents] = is_truncated_n
            info[agents] = info_n

        if is_terminal_n or is_truncated_n:
            self.agents = []

        return observations, rewards, terminations, truncations, info

    def map_local_actions_to_global_action(
        self, actions: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Maps multi agent actions into single agent action space.

        Args:
            action: An dict representing the action of each agent
        Returns:
            The action of the whole domain (is what eqivilent single agent action would be)

        Raises:
            AssertionError:
                If the Agent action factorization is badly defined (if an action is double defined or not defined at all)
        """
        if self.agent_obsk is None:
            return actions[self.possible_agents[0]]

        assert self.single_agent_env.action_space.shape is not None
        global_action = (
            np.zeros((self.single_agent_env.action_space.shape[0],)) + np.nan
        )
        for agent_id, partition in enumerate(self.agent_action_partitions):
            for act_index, body_part in enumerate(partition):
                assert np.isnan(
                    global_action[body_part.act_ids]
                ), "FATAL: At least one gym_env action is doubly defined!"
                global_action[body_part.act_ids] = actions[
                    self.possible_agents[agent_id]
                ][act_index]

        assert not np.isnan(
            global_action
        ).any(), "FATAL: At least one gym_env action is undefined!"
        return global_action

    def map_global_action_to_local_actions(
        self, action: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Maps single agent action into multi agent action spaces.

        Args:
            action: An array representing the actions of the single agent for this domain
        Returns:
            A dictionary of actions to be performed by each agent
        Raises:
            AssertionError:
                If the Agent action factorization sizes are badly defined
        """
        if self.agent_obsk is None:
            return {self.possible_agents[0]: action}

        local_actions = {}
        for agent_id, partition in enumerate(self.agent_action_partitions):
            local_actions[self.possible_agents[agent_id]] = np.array(
                [action[node.act_ids] for node in partition]
            )

        # assert sizes
        assert len(local_actions) == len(self.action_spaces)
        for agent in self.possible_agents:
            assert len(local_actions[agent]) == self.action_spaces[agent].shape[0]

        return local_actions

    def map_global_state_to_local_observations(
        self, global_state: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Maps single agent observation into multi agent observation spaces.

        Args:
            global_state:
                the global_state (generated from MaMuJoCo.state())

        Returns:
            A dictionary of states that would be observed by each agent given the 'global_state'
        """
        if self.agent_obsk is None:
            return {self.possible_agents[0]: global_state}

        class data_struct:
            def __init__(self, qpos, qvel, cinert, cvel, qfrc_actuator, cfrc_ext):
                self.qpos = qpos
                self.qvel = qvel
                self.cinert = cinert
                self.cvel = cvel
                self.qfrc_actuator = qfrc_actuator
                self.cfrc_ext = cfrc_ext

        obs_struct = _observation_structure(self.single_agent_env.spec.id)
        qpos_end_index = obs_struct["qpos"]
        qvel_end_index = qpos_end_index + obs_struct["qvel"]
        cinert_end_index = qvel_end_index + obs_struct["cinert"]
        cvel_end_index = cinert_end_index + obs_struct["cvel"]
        qfrc_actuator_end_index = cvel_end_index + obs_struct["qfrc_actuator"]
        cfrc_ext_end_index = qfrc_actuator_end_index + obs_struct["cfrc_ext"]

        assert len(global_state) == cfrc_ext_end_index

        data = data_struct(
            qpos=np.concatenate(
                (
                    np.zeros(obs_struct["skipped_qpos"]),
                    global_state[0:qpos_end_index],
                )
            ),
            qvel=np.array(global_state[qpos_end_index:qvel_end_index]),
            cinert=np.array(global_state[qvel_end_index:cinert_end_index]),
            cvel=np.array(global_state[cinert_end_index:cvel_end_index]),
            qfrc_actuator=np.array(
                global_state[cvel_end_index:qfrc_actuator_end_index]
            ),
            cfrc_ext=np.array(global_state[qfrc_actuator_end_index:cfrc_ext_end_index]),
        )

        if len(data.cinert) != 0:
            data.cinert = np.reshape(
                data.cinert, self.single_agent_env.unwrapped.data.cinert.shape
            )
        if len(data.cvel) != 0:
            data.cvel = np.reshape(
                data.cvel, self.single_agent_env.unwrapped.data.cvel.shape
            )
        if len(data.cfrc_ext) != 0:
            data.cfrc_ext = np.reshape(
                data.cfrc_ext, self.single_agent_env.unwrapped.data.cfrc_ext.shape
            )

        assert len(self.single_agent_env.unwrapped.data.qpos.flat) == len(data.qpos)
        assert len(self.single_agent_env.unwrapped.data.qvel.flat) == len(data.qvel)

        observations = {}
        for agent_id, agent in enumerate(self.possible_agents):
            observations[agent] = self._get_obs_agent(agent_id, data)
        return observations

    def map_local_observation_to_global_state(
        self, local_observations: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Maps multi agent observations into single agent observation space.

        NOT IMPLEMENTED, try using MaMuJoCo.state() instead

        Args:
            local_obserations:
                the local observation of each agents (generated from MaMuJoCo.step())

        Returns:
            the global observations that correspond to a single agent (what you would get with MaMuJoCo.state())
        """
        # Dev notes for anyone who attempts to implement it:
        # - Depending on the factorization the local observations may not observe the total global observable space, you will need to handle that
        raise NotImplementedError

    def observation_space(self, agent: str) -> gymnasium.spaces.Box:
        """See [pettingzoo.utils.env.ParallelEnv.observation_space](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.observation_space)."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Box:
        """See [pettingzoo.utils.env.ParallelEnv.action_space](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.action_space)."""
        return self.action_spaces[agent]

    def state(self) -> np.ndarray:
        """See [pettingzoo.utils.env.ParallelEnv.state](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.state)."""
        return self.single_agent_env.unwrapped._get_obs()

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Returns: all agent's observations in a dict[str, ActionType]."""
        # dev NOTE: ignores `self.single_agent_env._get_obs()` and builds observations using obsk.build_obs()
        observations = {}
        for agent_id, agent in enumerate(self.possible_agents):
            observations[agent] = self._get_obs_agent(agent_id)
        return observations

    def _get_obs_agent(self, agent_id: int, data=None) -> np.array:
        """Get the observation of single agent.

        Args:
            agent_id: The id in self.possible_agents.values()
            data: An optional overwrite of the MuJoCo data, defaults to the data at the current time step
        Returns:
            The observation of the agent given the data
        """
        if self.agent_obsk is None:
            return self.single_agent_env.unwrapped._get_obs()
        if data is None:
            data = self.single_agent_env.unwrapped.data

        return build_obs(
            data,
            self.k_dicts[agent_id],
            self.k_categories,
            self.mujoco_globals,
            self.global_categories,
        )

    def reset(self, seed=None, return_info=False, options=None):
        """Resets the the `single_agent_env`.

        Args:
            seed: see pettingzoo.utils.env.ParallelEnv.reset() doc
            return_info: see pettingzoo.utils.env.ParallelEnv.reset() doc
            options: Ignored arguments
        Returns:
            Initial observations and info
        """
        _, info_n = self.single_agent_env.reset(seed=seed)
        info = {}
        for agent in self.possible_agents:
            info[agent] = info_n
        self.agents = self.possible_agents
        if return_info is False:
            return self._get_obs()
        else:
            return self._get_obs(), info

    def render(self):
        """Renders the MuJoCo environment using the mechanism of the single agent Gymnasium-MuJoCo.

        Returns:
            The same return value as the single agent Gymnasium.MuJoCo
            see https://gymnasium.farama.org/environments/mujoco/
        """
        return self.single_agent_env.render()

    def close(self):
        """See [pettingzoo.utils.env.ParallelEnv.close](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.close)."""
        self.single_agent_env.close()

    def seed(self, seed: int = None):
        """Not implemented use env.reset(seed=...) instead."""
        raise NotImplementedError("use env.reset(seed=...) instead")

    def _generate_local_categories(self, scenario: str) -> list[list[str]]:
        """Generated the default observation categories for each observation depth.

        Args:
            scenario: the mujoco task
        Returns:
            a list of observetion types per observation depth
        """
        if self.agent_obsk is None:
            return None

        if scenario in ["Ant-v4", "manyagent_ant"]:
            # k_split = ["qpos,qvel,cfrc_ext", "qpos"]  # Gymansium.MuJoCo.Ant-v4 has disabled cfrc_ext by default
            k_split = ["qpos,qvel", "qpos"]
        elif scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
            k_split = [
                "qpos,qvel,cinert,cvel,qfrc_actuator,cfrc_ext",
                "qpos",
            ]
        elif scenario in ["coupled_half_cheetah-v4"]:
            k_split = ["qpos,qvel,ten_J,ten_length,ten_velocity", "qpos"]
        else:
            k_split = ["qpos,qvel", "qpos"]

        categories = [
            k_split[k if k < len(k_split) else -1].split(",")
            for k in range(self.agent_obsk + 1)
        ]
        return categories

    def _generate_global_categories(self, scenario: str) -> list[str]:
        """Generates the default global categories of observations.

        Args:
            scenario: The name of the MuJoCo Task
        Returns:
            The default Global Categories for the scenario (a list of all observable types for that domain)
        """
        if self.agent_obsk is None:
            return []

        if scenario in ["Ant-v4", "manyagent_ant"]:
            return ["qpos", "qvel"]
        elif scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
            return ["qpos", "qvel", "cinert", "cvel", "qfrc_actuator", "cfrc_ext"]
        elif scenario in ["coupled_half_cheetah-v4"]:
            return ["qpos", "qvel"]
        else:
            return ["qpos", "qvel"]
