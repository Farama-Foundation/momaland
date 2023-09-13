"""Item Gathering environment, based on the environment from the paper below.

Johan Källström and Fredrik Heintz. 2019. Tunable Dynamics in Agent-Based Simulation using Multi-Objective
Reinforcement Learning. Presented at the Adaptive and Learning Agents Workshop at AAMAS 2019.
https://liu.diva-portal.org/smash/record.jsf?pid=diva2%3A1362933&dswid=9018
"""

import functools
from copy import deepcopy

# from gymnasium.utils import EzPickle
from typing_extensions import override

import numpy as np
from gymnasium.logger import warn
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils import wrappers

from momadm_benchmarks.envs.item_gathering.map_utils import DEFAULT_MAP
from momadm_benchmarks.utils.conversions import mo_parallel_to_aec
from momadm_benchmarks.utils.env import MOParallelEnv


ACTIONS = {
    0: np.array([-1, 0], dtype=np.int32),  # up
    1: np.array([1, 0], dtype=np.int32),  # down
    2: np.array([0, -1], dtype=np.int32),  # left
    3: np.array([0, 1], dtype=np.int32),  # right
}


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
        env_map=None,
        render_mode=None,
    ):
        """Initializes the item gathering domain.

        Args:
            num_timesteps: number of timesteps to run the environment for
            env_map: map of the environment
            render_mode: render mode for the environment
        """
        self.num_timesteps = num_timesteps
        self.render_mode = render_mode

        if env_map is not None:
            self.env_map = env_map
        else:
            self.env_map = deepcopy(DEFAULT_MAP)

        # TODO check if the map is valid, e.g. there should be no #2, all values should be integers,
        #  objective values encodings should be sequential

        self.agent_positions = np.argwhere(self.env_map == 1)  # store agent positions in separate list
        self.env_map[self.env_map == 1] = 0  # remove agent starting positions from map

        self.possible_agents = ["agent_" + str(r) for r in range(len(self.agent_positions))]
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.action_spaces = dict(zip(self.agents, [Discrete(len(ACTIONS))] * len(self.agent_positions)))

        # observation space is a 2D array, the same size as the grid
        # 0 for empty, 1 for the current agent, 2 for other agents, 3 for objective 1, 4 for objective 2, ...
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Box(
                        low=0.0,
                        high=np.max(self.env_map),
                        shape=self.env_map.shape,
                        dtype=np.float32,
                    )
                ]
                * len(self.agent_positions),
            )
        )

        # determine the number of item types and maximum number of each item
        all_map_entries = np.unique(self.env_map, return_counts=True)
        indices_of_items = np.argwhere(all_map_entries[0] > 2).flatten()
        item_counts = np.take(all_map_entries[1], indices_of_items)

        self.reward_spaces = dict(
            zip(self.agents, [Box(low=0, high=max(item_counts), shape=(len(item_counts),))] * len(self.agent_positions))
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

    def _create_observation(self, agent_id):
        """Function to create the observation passed to each agent at the end of a timestep.

        Args:
            agent_id: the id of the agent

        Returns: a 2D Numpy array with the following items encoded:
        - 0 is empty space
        - 1 is the position of the agent with the specified agent_id
        - 2 is the position of any other agents
        - 3, 4, 5 ... denote the locations of items representing different objectives

        """
        obs = np.zeros((self.rows, self.columns))
        # TODO
        return obs
