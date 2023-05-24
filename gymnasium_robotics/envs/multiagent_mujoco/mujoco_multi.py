"""Main file for MaMuJoCo includes the MultiAgentMujocoEnv class.

This file is originally from the `schroederdewitt/multiagent_mujoco` repository hosted on GitHub
(https://github.com/schroederdewitt/multiagent_mujoco/blob/master/multiagent_mujoco/mujoco_multi.py)
Original Author: Schroeder de Witt

Then Modified by @Kallinteris-Andreas for this project
changes:
 - General code cleanup, factorization, type hinting, adding documentation and comments
 - Now uses PettingZoo APIs instead of a propriatery API
 - Now supports custom factorizations
 - Added new functions MultiAgentMujocoEnv.map_global_action_to_local_actions, MultiAgentMujocoEnv.map_local_actions_to_global_action, MultiAgentMujocoEnv.map_local_observation_to_global_state, MultiAgentMujocoEnv.map_local_observation_to_global_state

This project is covered by the Apache 2.0 License.
"""


from __future__ import annotations

import gymnasium
import numpy as np
import pettingzoo
from gymnasium.wrappers.time_limit import TimeLimit

from gymnasium_robotics.envs.multiagent_mujoco.coupled_half_cheetah import (
    CoupledHalfCheetahEnv,
)
from gymnasium_robotics.envs.multiagent_mujoco.many_segment_ant import ManySegmentAntEnv
from gymnasium_robotics.envs.multiagent_mujoco.many_segment_swimmer import (
    ManySegmentSwimmerEnv,
)
from gymnasium_robotics.envs.multiagent_mujoco.obsk import (
    Node,
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

    Doc can be found at (https://robotics.farama.org/envs/mamujoco/)
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
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        agent_factorization: dict | None = None,
        local_categories: list[list[str]] | None = None,
        global_categories: tuple[str, ...] | None = None,
        render_mode: str | None = None,
        **kwargs,
    ):
        """Init.

        Args:
            scenario: The Task/Environment, valid values:
                "Ant", "HalfCheetah", "Hopper", "HumanoidStandup", "Humanoid", "Reacher", "Swimmer", "Pusher", "Walker2d", "InvertedPendulum", "InvertedDoublePendulum", "ManySegmentSwimmer", "ManySegmentAnt", "CoupledHalfCheetah"
            agent_conf: '${Number Of Agents}x${Number Of Segments per Agent}${Optionally Additional options}', eg '1x6', '2x4', '2x4d',
                If it set to None the task becomes single agent (the agent observes the entire environment, and performs all the actions)
            agent_obsk: Number of nearest joints to observe,
                If set to 0 it only observes local state,
                If set to 1 it observes local state + 1 joint over,
                If set to 2 it observes local state + 2 joints over,
                If it set to None the task becomes single agent (the agent observes the entire environment, and performs all the actions)
                The Default value is: 1
            agent_factorization: A custom factorization of the MuJoCo environment (overwrites agent_conf),
                see DOC [how to create new agent factorizations](https://robotics.farama.org/envs/MaMuJoCo/index.html#how-to-create-new-agent-factorizations).
            local_categories: The categories of local observations for each observation depth,
                It takes the form of a list where the k-th element is the list of observable items observable at the k-th depth
                For example: if it is set to `[["qpos, qvel"], ["qvel"]]` then means each agent observes its own position and velocity elements, and it's neighbors velocity elements.
                The default is: Check each environment's page on the "observation space" section.
            global_categories: The categories of observations extracted from the global observable space,
                For example: if it is set to `("qpos")` out of the globally observable items of the environment, only the position items will be observed.
                The default is: Check each environment's page on the "observation space" section.
            render_mode: see [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/),
                valid values: 'human', 'rgb_array', 'depth_array'
            kwargs: Additional arguments passed to the [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/) environment,
                Note: arguments that change the observation space will not work.

            Raises: NotImplementedError: When the scenario is not supported (not part of of the valid values)
        """
        scenario += "-v4"

        # load the underlying single agent Gymansium MuJoCo Environment in `self.single_agent_env`
        if scenario in _MUJOCO_GYM_ENVIROMENTS:
            self.single_agent_env: gymnasium.envs.mujoco.mujoco_env.MujocoEnv = (
                gymnasium.make(scenario, **kwargs, render_mode=render_mode)
            )
        elif scenario in ["ManySegmentAnt-v4"]:
            assert isinstance(agent_conf, str)
            try:
                n_segs = int(agent_conf.split("x")[0]) * int(agent_conf.split("x")[1])
            except Exception:
                raise Exception(f"UNKNOWN partitioning config: {agent_conf}")

            self.single_agent_env = TimeLimit(
                ManySegmentAntEnv(n_segs, render_mode), max_episode_steps=1000
            )
        elif scenario in ["ManySegmentSwimmer-v4"]:
            assert isinstance(agent_conf, str)
            try:
                n_segs = int(agent_conf.split("x")[0]) * int(agent_conf.split("x")[1])
            except Exception:
                raise Exception(f"UNKNOWN partitioning config: {agent_conf}")

            self.single_agent_env = TimeLimit(
                ManySegmentSwimmerEnv(n_segs, render_mode), max_episode_steps=1000
            )
        elif scenario in ["CoupledHalfCheetah-v4"]:
            self.single_agent_env = TimeLimit(
                CoupledHalfCheetahEnv(render_mode), max_episode_steps=1000
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
            dummy_node = Node("dummy_node", None, None, None)
            self.agent_action_partitions = [
                tuple(
                    dummy_node
                    for i in range(self.single_agent_env.action_space.shape[0])
                )
            ]
            mujoco_edges = []

        # Create agent lists
        self.possible_agents = [
            "agent_" + str(agent_id)
            for agent_id in range(len(self.agent_action_partitions))
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
        if self.agent_obsk is not None:
            self.k_dicts = [
                get_joints_at_kdist(
                    self.agent_action_partitions[agent_id],
                    mujoco_edges,
                    k=self.agent_obsk,
                )
                for agent_id in range(self.num_agents)
            ]

        # Create observation and action spaces
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

        mujoco_data = data_struct(
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

        if len(mujoco_data.cinert) != 0:
            mujoco_data.cinert = np.reshape(
                mujoco_data.cinert, self.single_agent_env.data.cinert.shape
            )
        if len(mujoco_data.cvel) != 0:
            mujoco_data.cvel = np.reshape(
                mujoco_data.cvel, self.single_agent_env.data.cvel.shape
            )
        if len(mujoco_data.cfrc_ext) != 0:
            mujoco_data.cfrc_ext = np.reshape(
                mujoco_data.cfrc_ext, self.single_agent_env.data.cfrc_ext.shape
            )

        assert len(self.single_agent_env.unwrapped.data.qpos.flat) == len(
            mujoco_data.qpos
        )
        assert len(self.single_agent_env.unwrapped.data.qvel.flat) == len(
            mujoco_data.qvel
        )

        observations = {}
        for agent_id, agent in enumerate(self.possible_agents):
            observations[agent] = self._get_obs_agent(agent_id, mujoco_data)
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

    def _get_obs_agent(self, agent_id: int, data=None) -> np.ndarray:
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
            data = self.single_agent_env.data

        return build_obs(
            data,
            self.k_dicts[agent_id],
            self.k_categories,
            self.mujoco_globals,
            self.global_categories,
        )

    def reset(self, seed: int | None = None, options=None):
        """Resets the the `single_agent_env`.

        Args:
            seed: see [pettingzoo.utils.env.ParallelEnv.reset()](https://pettingzoo.farama.org/api/parallel/#pettingzoo.utils.env.ParallelEnv.reset) doc
            options: Ignored arguments

        Returns:
            Initial observations and info
        """
        _, info_n = self.single_agent_env.reset(seed=seed)
        info = {}
        for agent in self.possible_agents:
            info[agent] = info_n
        self.agents = self.possible_agents
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

    def seed(self, seed: int | None = None):
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
            return [[]]

        if scenario in ["Ant-v4", "ManySegmentAnt"]:
            # k_split = ["qpos,qvel,cfrc_ext", "qpos"]  # Gymansium.MuJoCo.Ant-v4 has disabled cfrc_ext by default
            k_split = ["qpos,qvel", "qpos"]
        elif scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
            k_split = [
                "qpos,qvel,cinert,cvel,qfrc_actuator,cfrc_ext",
                "qpos",
            ]
        elif scenario in ["CoupledHalfCheetah-v4"]:
            k_split = ["qpos,qvel,ten_J,ten_length,ten_velocity", "qpos"]
        elif scenario in ["Reacher-v4"]:
            k_split = ["qpos,qvel,fingertip_dist", "qpos"]
        else:
            k_split = ["qpos,qvel", "qpos"]

        categories = [
            k_split[k if k < len(k_split) else -1].split(",")
            for k in range(self.agent_obsk + 1)
        ]
        return categories

    def _generate_global_categories(self, scenario: str) -> tuple[str, ...]:
        """Generates the default global categories of observations.

        Args:
            scenario: The name of the MuJoCo Task

        Returns:
            The default Global Categories for the scenario (a list of all observable types for that domain)
        """
        if self.agent_obsk is None:
            return ()

        if scenario in ["Ant-v4", "ManySegmentAnt"]:
            return ("qpos", "qvel")
        elif scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
            return ("qpos", "qvel", "cinert", "cvel", "qfrc_actuator", "cfrc_ext")
        elif scenario in ["CoupledHalfCheetah-v4"]:
            return ("qpos", "qvel")
        else:
            return ("qpos", "qvel")


# These are the export functions (for `PettingZoo` style exportations)
env = pettingzoo.utils.conversions.aec_wrapper_fn(MultiAgentMujocoEnv)
parallel_env = MultiAgentMujocoEnv
raw_parallel_env = MultiAgentMujocoEnv
