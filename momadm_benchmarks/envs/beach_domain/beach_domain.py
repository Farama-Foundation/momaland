"""Beach problem domain.

From Mannion, P., Devlin, S., Duggan, J., and Howley, E. (2018). Reward shaping for knowledge-based multi-objective multi-agent reinforcement learning.
"""

import functools
import random

# from gymnasium.utils import EzPickle
from typing_extensions import override

import numpy as np
from gymnasium.logger import warn
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


LEFT = -1
RIGHT = 1
STAY = 0
MOVES = ["LEFT", "RIGHT", "STAY"]
NUM_OBJECTIVES = 2


def parallel_env(**kwargs):
    return MOBeach(**kwargs)


def env(**kwargs):
    """Autowrapper for the beach domain.

    Args:
        **kwargs: keyword args to forward to the raw_env function

    Returns:
        A fully wrapped env
    """
    env = raw_env(**kwargs)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(**kwargs):
    """To support the AEC API, the raw_env function just uses the from_parallel function to convert from a ParallelEnv to an AEC env."""
    env = parallel_env(**kwargs)
    env = parallel_to_aec(env)
    return env


class MOBeach(ParallelEnv):
    """Environment for MO Beach problem domain.

    The init method takes in environment arguments and should define the following attributes:
    - possible_agents
    - action_spaces
    - observation_spaces
    These attributes should not be changed after initialization.
    """

    metadata = {"render_modes": ["human"], "name": "mobeach_v0"}

    def __init__(
        self,
        num_timesteps=10,
        num_agents=100,
        reward_scheme="local",
        sections=3,
        capacity=10,
        type_distribution=(0.5, 0.5),
        position_distribution=(0.5, 0.5, 1),
        render_mode=None,
    ):
        """Initializes the beach domain.

        Args:
            sections: TODO
            capacity: TODO
            num_agents: number of agents in the domain
            type_distribution: TODO # assume that there is an even mixing between two agent types unless specified
            position_distribution: TODO # assume that agents ae evenly distributed among sections unless specified
            num_timesteps: TODO
            render_mode: render mode
            reward_scheme: TODO # global or local rewards
        """
        self.reward_scheme = reward_scheme
        self.sections = sections
        # TODO Extend to distinct capacities per section?
        self.resource_capacities = [capacity for _ in range(sections)]
        self.num_timesteps = num_timesteps
        self.episode_num = 0
        self.type_distribution = type_distribution
        assert (
            len(position_distribution) == self.sections
        ), "number of sections should be equal to the length of the provided position_distribution:"
        self.position_distribution = position_distribution

        self.render_mode = render_mode
        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.types, self.state = self.init_state()

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
                        shape=(1, 5),
                    )
                ]
                * num_agents,
            )
        )
        # TODO check reward spaces
        self.reward_spaces = dict(zip(self.agents, [Box(low=0, high=1, shape=(2,))] * num_agents))

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

    def reward_space(self, agent):
        """Returns the reward space for the given agent."""
        return self.reward_spaces[agent]

    def render(self):
        """Renders the environment.

        In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return

    def close(self):
        """Close should release any graphical displays, subprocesses, network connections or any other environment data which should not be kept around after the user is no longer using the environment."""
        pass

    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.types, self.state = self.init_state()
        observations = {agent: None for agent in self.agents}
        self.episode_num = 0

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def init_state(self):
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
            act = actions[i]
            self.state[i] = min(self.sections - 1, max(self.state[i] + act, 0))

        section_consumptions = np.zeros(self.sections)
        section_agent_types = np.zeros((self.sections, len(self.type_distribution)))

        for i in range(len(self.agents)):
            section_consumptions[self.state[i]] += 1
            print(i, self.state[i], self.types[i])
            section_agent_types[self.state[i]][self.types[i]] += 1

        # print(section_agent_types)
        self.episode_num += 1

        env_termination = self.episode_num >= self.num_timesteps
        reward_per_section = np.zeros((self.sections, NUM_OBJECTIVES))

        if env_termination:
            if self.reward_scheme == "local":
                for i in range(self.sections):
                    lr_capacity = local_capacity_reward(self.resource_capacities[i], section_consumptions[i])
                    lr_mixture = local_mixture_reward(section_agent_types[i])
                    reward_per_section[i] = np.array([lr_capacity, lr_mixture])

            elif self.reward_scheme == "global":
                g_capacity = global_capacity_reward(self.resource_capacities, section_consumptions)
                g_mixture = global_mixture_reward(section_agent_types)
                reward_per_section = np.array([[g_capacity, g_mixture]] * self.sections)
                print("reward_per_section", reward_per_section)

        # Obs: agent type, section id, section capacity, section consumption, % of agents of current type
        observations = {agent: None for agent in self.agents}
        # Note that agents only receive the reward after the last timestep
        rewards = {self.agents[i]: [0, 0] for _ in range(self.num_agents)}

        for i, agent in enumerate(self.agents):
            total_same_type = section_agent_types[self.state[i]][self.types[i]]
            t = total_same_type / section_consumptions[self.state[i]]
            obs = [
                self.types[i],
                self.state[i],
                self.resource_capacities[self.state[i]],
                section_consumptions[self.state[i]],
                t,
            ]
            observations[agent] = obs
            rewards[agent] = reward_per_section[self.state[i]]

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_termination:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, env_termination, infos


def global_capacity_reward(capacities, consumptions):
    global_capacity_r = 0
    for i in range(len(capacities)):
        global_capacity_r += local_capacity_reward(capacities[i], consumptions[i])
    return global_capacity_r


def local_capacity_reward(capacity, consumption):
    # TODO make capacity lookup table to save CPU!
    return consumption * np.exp(-consumption / capacity)


def global_mixture_reward(section_agent_types):
    sum_local_mix = 0
    for i in range(len(section_agent_types)):
        sum_local_mix += local_mixture_reward(section_agent_types[i])
    return sum_local_mix / len(section_agent_types)


def local_mixture_reward(types):
    lr_mixture = 0
    if sum(types) > 0:
        lr_mixture = min(types) / sum(types)
    return lr_mixture
