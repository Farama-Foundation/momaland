"""Item Gathering environment, based on the environment from the paper below.

Johan Källström and Fredrik Heintz. 2019. Tunable Dynamics in Agent-Based Simulation using Multi-Objective
Reinforcement Learning. Presented at the Adaptive and Learning Agents Workshop at AAMAS 2019.
https://liu.diva-portal.org/smash/record.jsf?pid=diva2%3A1362933&dswid=9018
"""

import functools

# from gymnasium.utils import EzPickle
from typing_extensions import override

import numpy as np
from gymnasium.logger import warn
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils import wrappers

from momadm_benchmarks.utils.conversions import mo_parallel_to_aec
from momadm_benchmarks.utils.env import MOParallelEnv


def parallel_env(**kwargs):
    """Env factory function for the beach domain."""
    return MOItemGathering(**kwargs)


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
    env = mo_parallel_to_aec(env)
    return env


class MOItemGathering(MOParallelEnv):
    """Environment for the Item Gathering domain.

    The init method takes in environment arguments and should define the following attributes:
    - possible_agents
    - action_spaces
    - observation_spaces
    These attributes should not be changed after initialization.
    """

    metadata = {"render_modes": ["human"], "name": "iteamgathering_v0"}

    def __init__(
        self,
        num_timesteps=10,
        num_agents=2,
        rows=6,
        columns=6,
        item_distribution=(3, 3, 2),  # red, green, yellow
        item_locations="fixed",
        agent_locations="fixed",
        map=None,
        render_mode=None,
    ):
        """Initializes the beach domain.

        Args:
            num_timesteps: number of timesteps to run the environment for
            num_agents: number of agents in the environment
            rows: number of rows in the grid
            columns: number of columns in the grid
            item_distribution: distribution of items in the environment
            item_locations: location of the item to be gathered
            agent_locations: starting locations of the agent
            map: map of the environment
            render_mode: render mode for the environment
        """
        self.num_timesteps = num_timesteps
        self.num_agents = num_agents
        self.rows = rows
        self.columns = columns
        self.item_distribution = item_distribution
        self.item_locations = item_locations
        self.agent_locations = agent_locations
        self.render_mode = render_mode
        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        if map is not None:
            self.map = map
        else:
            self.map = np.zeros((self.rows, self.columns), dtype=np.int32)

        self.dir = {
            0: np.array([-1, 0], dtype=np.int32),  # up
            1: np.array([1, 0], dtype=np.int32),  # down
            2: np.array([0, -1], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32),  # right
        }

        self.action_spaces = dict(zip(self.agents, [Discrete(len(self.dir))] * num_agents))
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Box(
                        low=0.0,
                        high=5.0,
                        # Observation form:
                        # rows x columns matrix with encoded values for agents and items
                        shape=(
                            self.rows,
                            self.columns,
                        ),
                        dtype=np.float32,
                    )
                ]
                * num_agents,
            )
        )

        self.reward_spaces = dict(
            zip(self.agents, [Box(low=0, high=max(self.item_distribution), shape=(len(self.item_distribution),))] * num_agents)
        )

    def _init_state(self):
        """Initializes the state of the environment. This is called by reset()."""
        pass

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
        pass

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
        pass
