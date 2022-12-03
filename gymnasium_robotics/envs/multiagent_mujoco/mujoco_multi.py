import gymnasium
import numpy
import pettingzoo

from .coupled_half_cheetah import CoupledHalfCheetah
from .manyagent_ant import ManyAgentAntEnv
from .manyagent_swimmer import ManyAgentSwimmerEnv
from .obsk import build_obs, get_joints_at_kdist, get_parts_and_edges


_MUJOCO_GYM_ENVIROMENTS = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "HumanoidStandup-v4",
    "Humanoid-v4",
    "Reacher-v4",
    "Swimmer-v4",
    "Walker2d-v4",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
]


class MaMuJoCo(pettingzoo.utils.env.ParallelEnv):
    """
    # MaMuJoCo (Multi-Agent MuJoCo)

    These environments were introduced in ["FACMAC: Factored Multi-Agent Centralised Policy Gradients"](https://arxiv.org/abs/2003.06709)

    There are 2 types of Environments included (1) multi-agent factorizations of [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/) tasks and (2) new complex MuJoCo tasks meant to me solved with multi-agent Algorithms

    This Represents the first easy to use Frameworks for research of agent factorization

    # Action Spaces

    For (1) the action space shape is the shape of the single agent domain divided by the number of agents

    For (2) it Depends on the configuration

    # State Spaces

    Depends on Environment

    # valid Configurations

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

    """

    def __init__(
        self,
        scenario: str,
        agent_conf: str,
        agent_obsk: int = 1,
        render_mode: str = None,
    ):
        """
        Arguments:
            scenario: The Task to solve
            agent_conf: '${Number Of Agents}x${Number Of Segments per Agent}${Optionally Additional options}', eg '1x6', '2x4', '2x4d', if it set to None the task becomes single agent (the agent observes the entire environment, and performs all the actions)
            agent_obsk: Number of nearest joints to observe, if set to 0 it only observes local state, if set to 1 it observes local state + 1 joint over, if it set to None the task becomes single agent (the agent observes the entire environment, and performs all the actions)
            render_mode: see [Gymansium/MuJoCo](https://gymnasium.farama.org/environments/mujoco/), valid values: 'human', 'rgb_array', 'depth_array'
        """
        scenario += "-v4"
        self.global_categories = []

        # load scenario from script
        if scenario in _MUJOCO_GYM_ENVIROMENTS:
            self.env = gymnasium.make(scenario, render_mode=render_mode)
        elif scenario in ["manyagent_ant-v4"]:
            self.env = gymnasium.wrappers.TimeLimit(
                ManyAgentAntEnv(agent_conf, render_mode), max_episode_steps=1000
            )
        elif scenario in ["manyagent_swimmer-v4"]:
            self.env = gymnasium.wrappers.TimeLimit(
                ManyAgentSwimmerEnv(agent_conf, render_mode), max_episode_steps=1000
            )
        elif scenario in ["coupled_half_cheetah-v4"]:
            self.env = gymnasium.wrappers.TimeLimit(
                CoupledHalfCheetah(agent_conf, render_mode), max_episode_steps=1000
            )
        else:
            raise NotImplementedError("Custom env not implemented!")

        if agent_conf is None:
            self.agent_obsk = None
        else:
            self.agent_obsk = agent_obsk  # if None, fully observable else k>=0 implies observe nearest k agents or joints

        if self.agent_obsk is not None:
            (
                self.agent_action_partitions,
                mujoco_edges,
                self.mujoco_globals,
            ) = get_parts_and_edges(scenario, agent_conf)
        else:
            self.agent_action_partitions = {
                "single agent": [
                    "action" + str(action_id)
                    for action_id in range(self.env.action_space.shape[0])
                ]
            }

        self.possible_agents = [
            str(agent_id) for agent_id in range(len(self.agent_action_partitions))
        ]
        self.agents = self.possible_agents

        self.k_categories = self._generate_categories(scenario)

        if self.agent_obsk is not None:
            self.k_dicts = [
                get_joints_at_kdist(
                    self.agent_action_partitions[agent_id],
                    mujoco_edges,
                    k=self.agent_obsk,
                )
                for agent_id in range(self.num_agents)
            ]

        if self.agent_obsk is None:
            self.action_spaces = {self.possible_agents[0]: self.env.action_space}
            self.observation_spaces = {
                self.possible_agents[0]: self.env.observation_space
            }
        else:
            self.observation_spaces, self.action_spaces = {}, {}
            for agent_id, partition in enumerate(self.agent_action_partitions):
                self.action_spaces[str(agent_id)] = gymnasium.spaces.Box(
                    low=self.env.action_space.low[0],
                    high=self.env.action_space.high[0],
                    shape=(len(partition),),
                    dtype=numpy.float32,
                )
                self.observation_spaces[str(agent_id)] = gymnasium.spaces.Box(
                    low=-numpy.inf,
                    high=numpy.inf,
                    shape=(len(self._get_obs_agent(agent_id)),),
                    dtype=numpy.float32,
                )

        pass

    def step(
        self, actions: dict[str, numpy.array]
    ) -> tuple[
        dict[str, numpy.array],
        dict[str, numpy.array],
        dict[str, numpy.array],
        dict[str, numpy.array],
        dict[str, str],
    ]:
        _, reward_n, is_terminal_n, is_truncated_n, info_n = self.env.step(
            self.map_local_actions_to_global_action(actions)
        )

        rewards, terminations, truncations, info = {}, {}, {}, {}
        observations = self._get_obs()
        for agent_id in self.agents:
            rewards[str(agent_id)] = reward_n
            terminations[str(agent_id)] = is_terminal_n
            truncations[str(agent_id)] = is_truncated_n
            info[str(agent_id)] = info_n

        if is_terminal_n or is_truncated_n:
            self.agents = []

        return observations, rewards, terminations, truncations, info

    def map_local_actions_to_global_action(
        self, actions: dict[str, numpy.array]
    ) -> numpy.array:
        """
        Maps actions back into MuJoCo action space
        Returns:
            The actions of the whole domain in a single list
        """
        if self.agent_obsk is None:
            return actions[self.possible_agents[0]]

        env_actions = numpy.zeros((self.env.action_space.shape[0],)) + numpy.nan
        for agent_id, partition in enumerate(self.agent_action_partitions):
            for i, body_part in enumerate(partition):
                assert numpy.isnan(
                    env_actions[body_part.act_ids]
                ), "FATAL: At least one env action is doubly defined!"
                env_actions[body_part.act_ids] = actions[str(agent_id)][i]

        assert not numpy.isnan(
            env_actions
        ).any(), "FATAL: At least one env action is undefined!"
        return env_actions

    def map_global_action_to_local_actions(
        self, action: numpy.ndarray
    ) -> dict[str, numpy.ndarray]:
        """
        Arguments:
            action: An array representing the actions of the single agent for this domain
        Returns:
            A dictionary of actions to be performed by each agent
        """
        if self.agent_obsk is None:
            return {self.possible_agents[0]: action}

        local_actions = {}
        for agent_id, partition in enumerate(self.agent_action_partitions):
            local_actions[self.possible_agents[agent_id]] = numpy.array(
                [action[node.act_ids] for node in partition]
            )

        # assert sizes
        assert len(local_actions) == len(self.action_spaces)
        for agent in self.possible_agents:
            assert len(local_actions[agent]) == self.action_spaces[agent_id].shape[0]

        return local_actions

    def map_global_state_to_local_observations(
        self, global_state: numpy.ndarray
    ) -> dict[str, numpy.ndarray]:
        # self.env.unwrapped
        # breakpoint()
        pass

    def observation_space(self, agent: str) -> gymnasium.spaces.Box:
        return self.observation_spaces[str(agent)]

    def action_space(self, agent: str) -> gymnasium.spaces.Box:
        return self.action_spaces[str(agent)]

    def state(self) -> numpy.array:
        return self.env.unwrapped._get_obs()

    def _get_obs(self) -> dict[str, numpy.array]:
        "Returns all agent observations in a dict[str, ActionType]"
        observations = {}
        for agent_id in self.agents:
            observations[str(agent_id)] = self._get_obs_agent(int(agent_id))
        return observations

    def _get_obs_agent(self, agent_id) -> numpy.array:
        if self.agent_obsk is None:
            return self.env.unwrapped._get_obs()
        else:
            return build_obs(
                self.env,
                self.k_dicts[agent_id],
                self.k_categories,
                self.mujoco_globals,
                self.global_categories,
            )

    def reset(self, seed=None, return_info=False, options=None):
        """Returns initial observations and states"""
        _, info_n = self.env.reset(seed=seed)
        info = {}
        for agent_id in self.agents:
            info[str(agent_id)] = info_n
        self.agents = self.possible_agents
        if return_info is False:
            return self._get_obs()
        else:
            return self._get_obs(), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed: int = None):
        raise NotImplementedError

    def _generate_categories(self, scenario: str):
        if self.agent_obsk is None:
            return None

        if scenario in ["Ant-v4", "manyagent_ant"]:
            k_split = ["qpos,qvel,cfrc_ext", "qpos"]
        elif scenario in ["Humanoid-v4", "HumanoidStandup-v4"]:
            k_split = ["qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator", "qpos"]
        elif scenario in ["Reacher-v4"]:
            k_split = ["qpos,qvel,fingertip_dist", "qpos"]
        elif scenario in ["coupled_half_cheetah-v4"]:
            k_split = ["qpos,qvel,ten_J,ten_length,ten_velocity", ""]
        else:
            k_split = ["qpos,qvel", "qpos"]

        categories = [
            k_split[k if k < len(k_split) else -1].split(",")
            for k in range(self.agent_obsk + 1)
        ]
        return categories
