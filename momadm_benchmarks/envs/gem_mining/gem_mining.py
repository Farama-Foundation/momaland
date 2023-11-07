"""Gem Mining Problem.

Combining the Mining Day environment from:
    Diederik M. Roijers - Multi-Objective Decision-Theoretic Planning, PhD Thesis, University of Amsterdam, 2016. (Which contains the multi-objective planning version.)
And the Gem Mining problem from:
    Eugenio Bargiacchi, Timothy Verstraeten, Diederik M. Roijers, Ann Nowé and Hado van Hasselt - Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems. In ICML 2018, Stockholm, July 2018. (Which contains the single-objective learning version.)

Created on Tue Oct 24 16:31:14 2023

@author: dmroijers
"""

import functools
import random

# from gymnasium.utils import EzPickle
from typing_extensions import override

import numpy as np
from gymnasium.logger import warn
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils import wrappers

from momadm_benchmarks.utils.conversions import mo_parallel_to_aec
from momadm_benchmarks.utils.env import MOParallelEnv


LEFT = -1
RIGHT = 1
STAY = 0
MOVES = ["LEFT", "RIGHT", "STAY"]
NUM_OBJECTIVES = 2


def parallel_env(**kwargs):
    """Parallel env factory function for the gem mining domain."""
    return raw_env(**kwargs)


def env(**kwargs):
    """Autowrapper for the gem mining domain.

    Args:
        **kwargs: keyword args to forward to the parallel_env function

    Returns:
        A fully wrapped env
    """
    env = parallel_env(**kwargs)
    env = mo_parallel_to_aec(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


def raw_env(**kwargs):
    """Env factory function for the gem mining domain."""
    return MOGemMining(**kwargs)


class MOGemMining(MOParallelEnv):
    """Environment for MO gem mining domain.

    The init method takes in environment arguments and should define the following attributes:
    - possible_agents
    - action_spaces
    - observation_spaces
    These attributes should not be changed after initialization.
    """

    metadata = {"render_modes": ["human"], "name": "mogem_v0"}

    # TODO does this environment require max_cycle?
    def __init__(
        self,
        num_timesteps=10,
        num_agents=100,
        reward_scheme="local",
        sections=6,
        capacity=10,
        type_distribution=(0.5, 0.5),
        position_distribution=None,
        render_mode=None,
    ):
        """Initializes the beach domain.

        Args:
            sections: number of beach sections in the domain
            capacity: capacity of each beach section
            num_agents: number of agents in the domain
            type_distribution: the distribution of agent types in the domain. Default: 2 types equally distributed.
            position_distribution: the initial distribution of agents in the domain. Default: uniform over all sections.
            num_timesteps: number of timesteps in the domain
            render_mode: render mode
            reward_scheme: the reward scheme to use ('local', or 'global'). Default: local
        """
        self.reward_scheme = reward_scheme
        self.sections = sections
        # TODO Extend to distinct capacities per section?
        self.resource_capacities = [capacity for _ in range(sections)]
        self.num_timesteps = num_timesteps
        self.episode_num = 0
        self.type_distribution = type_distribution
        if position_distribution is None:
            self.position_distribution = [1 / sections for _ in range(sections)]
        else:
            assert (
                len(position_distribution) == self.sections
            ), "number of sections should be equal to the length of the provided position_distribution:"
            self.position_distribution = position_distribution

        self.render_mode = render_mode
        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.types, self.state = self._init_state()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.action_spaces = dict(zip(self.agents, [Discrete(len(MOVES))] * num_agents))
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Box(
                        low=0,
                        high=self.num_agents,
                        # Observation form:
                        # agent type, section id, section capacity, section consumption, % of agents of current type
                        shape=(5,),
                        dtype=np.float32,
                    )
                ]
                * num_agents,
            )
        )

        # maximum capacity reward can be calculated  by calling the _global_capacity_reward()
        optimal_consumption = [capacity for _ in range(sections)]
        optimal_consumption[-1] = max(self.num_agents - ((sections - 1) * capacity), 0)
        # max_r = _global_capacity_reward(self.resource_capacities, optimal_consumption)
        self.reward_spaces = dict(zip(self.agents, [Box(low=0, high=10, shape=(NUM_OBJECTIVES,))] * num_agents))

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasiuspspom.farama.org/api/spaces/
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    @override
    def action_space(self, agent):
        return self.action_spaces[agent]

    @override
    def reward_space(self, agent):
        """Returns the reward space for the given agent."""
        return self.reward_spaces[agent]

    @override
    def render(self):
        """Renders the environment.

        In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return

    @override
    def close(self):
        """Close should release any graphical displays, subprocesses, network connections or any other environment data which should not be kept around after the user is no longer using the environment."""
        pass

    @override
    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.agents = self.possible_agents[:]
        self.types, self.state = self._init_state()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        section_consumptions, section_agent_types = self._get_stats()
        observations = {
            agent: self._get_obs(i, section_consumptions, section_agent_types) for i, agent in enumerate(self.agents)
        }
        self.episode_num = 0

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _init_state(self):
        """Initializes the state of the environment. This is called by reset()."""
        types = random.choices(
            [i for i in range(len(self.type_distribution))], weights=self.type_distribution, k=self.num_agents
        )

        if self.position_distribution is None:
            positions = [random.randint(0, self.sections - 1) for _ in self.agents]
        else:
            positions = random.choices(
                [i for i in range(self.sections)], weights=self.position_distribution, k=self.num_agents
            )
        return types, positions

    def step(self, actions):
        """Steps in the environment.

        Args:
            actions: a dict of actions, keyed by agent names

        Returns: a tuple containing the following items in order:
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Apply actions and update system state
        for i, agent in enumerate(self.agents):
            act = actions[agent]
            self.state[i] = min(self.sections - 1, max(self.state[i] + act, 0))

        section_consumptions, section_agent_types = self._get_stats()

        self.episode_num += 1

        env_termination = self.episode_num >= self.num_timesteps
        self.terminations = {agent: env_termination for agent in self.agents}
        reward_per_section = np.zeros((self.sections, NUM_OBJECTIVES), dtype=np.float32)

        if env_termination:
            if self.reward_scheme == "local":
                for i in range(self.sections):
                    lr_capacity = 10  # _local_capacity_reward(self.resource_capacities[i], section_consumptions[i])
                    lr_mixture = 10  # _local_mixture_reward(section_agent_types[i])
                    reward_per_section[i] = np.array([lr_capacity, lr_mixture])

            elif self.reward_scheme == "global":
                g_capacity = 10  # _global_capacity_reward(self.resource_capacities, section_consumptions)
                g_mixture = 10  # _global_mixture_reward(section_agent_types)
                reward_per_section = np.array([[g_capacity, g_mixture]] * self.sections)

        # Obs: agent type, section id, section capacity, section consumption, % of agents of current type
        observations = {agent: None for agent in self.agents}
        # Note that agents only receive the reward after the last timestep
        rewards = {self.agents[i]: np.array([0, 0], dtype=np.float32) for _ in range(self.num_agents)}

        for i, agent in enumerate(self.agents):
            observations[agent] = self._get_obs(i, section_consumptions, section_agent_types)
            rewards[agent] = reward_per_section[self.state[i]]

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_termination:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, self.truncations, self.terminations, infos

    def _get_obs(self, i, section_consumptions, section_agent_types):
        total_same_type = section_agent_types[self.state[i]][self.types[i]]
        t = total_same_type / section_consumptions[self.state[i]]
        obs = np.array(
            [self.types[i], self.state[i], self.resource_capacities[self.state[i]], section_consumptions[self.state[i]], t],
            dtype=np.float32,
        )
        return obs

    def _get_stats(self):
        section_consumptions = np.zeros(self.sections)
        section_agent_types = np.zeros((self.sections, len(self.type_distribution)))

        for i in range(len(self.agents)):
            section_consumptions[self.state[i]] += 1
            section_agent_types[self.state[i]][self.types[i]] += 1
        return section_consumptions, section_agent_types
