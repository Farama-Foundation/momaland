"""Item Gathering environment, based on the environment from the paper below.

Johan Källström and Fredrik Heintz. 2019. Tunable Dynamics in Agent-Based Simulation using Multi-Objective
Reinforcement Learning. Presented at the Adaptive and Learning Agents Workshop at AAMAS 2019.
https://liu.diva-portal.org/smash/record.jsf?pid=diva2%3A1362933&dswid=9018
"""

import functools
import random
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
        initial_map=None,
        render_mode=None,
    ):
        """Initializes the item gathering domain.

        Args:
            num_timesteps: number of timesteps to run the environment for
            initial_map: map of the environment
            render_mode: render mode for the environment
        """
        self.num_timesteps = num_timesteps
        self.current_timestep = 0
        self.render_mode = render_mode

        if initial_map is not None:
            self.initial_map = initial_map
        else:
            self.initial_map = DEFAULT_MAP

        # self.env_map is the working copy used in each episode. self.initial_map should not be modified.
        self.env_map = deepcopy(self.initial_map)

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
        """Close should release any graphical displays, subprocesses, network connections or any other environment data
        which should not be kept around after the user is no longer using the environment."""
        pass

    @override
    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        """
        if seed is not None:  # TODO Decide whether we need the seed
            np.random.seed(seed)
            random.seed(seed)
        self.agents = self.possible_agents[:]
        print("self.agents", self.agents)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.env_map = deepcopy(self.initial_map)  # Reset the environment map to the initial map provided
        self.agent_positions = np.argwhere(self.env_map == 1)  # store agent positions in separate list
        self.env_map[self.env_map == 1] = 0  # remove agent starting positions from map

        observations = {agent: self._create_observation(i) for i, agent in enumerate(self.agents)}
        self.time_num = 0

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def is_valid_position(self, position):
        """Check if the new position of an agent is valid.

        Args:
            position: the new position [x, y]

        Returns: a boolean indicating whether the new position is valid

        """
        row_valid = 0 <= position[0] < self.env_map.position[0]
        col_valid = 0 <= position[1] < self.env_map.position[1]
        return row_valid and col_valid

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
        pass  # implemented with pass for now, the commented code below is an initial rough attempt at implementation

        """
        # what happens if two agents want to move onto the same item? randomly allocate the item randomly
        # what happens if two agents want the same square? one of the agents is chosen at random to move onto it, the other one stays where it is
        # do items reappear? after 1 timestep? After multiple timesteps
        # environment termination, after all timesteps are exhausted or all items or gathered

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        new_positions = deepcopy(self.agent_positions)

        # Apply actions and update system state
        # TODO test this
        for i, agent in enumerate(self.agents):
            act = actions[agent]
            new_position = self.agent_positions[i] + ACTIONS[act]
            if self.is_valid_position(new_position):
                new_positions[i] = new_position

        # check for collisions, resolve here only the collision, have final new position list at end
        # for i in range(len(new_positions)):
        #     for j in range(i+1, len(new_positions)):
        #         elif new_positions[]

        # create empty reward vectors - figure out later
        # rewards = {self.agents[i]: np.array([0, 0], dtype=np.float32) for _ in range(self.num_agents)}
        temp_rewards = []  # TODO for now, need to change
        for i in range(len(new_positions)):
            temp_rewards.append([0, 0, 0])

        # update all reward vectors with collected items (if any), delete items from the map
        for i in range(len(new_positions)):
            value_in_cell = self.env_map[new_positions[i][0], new_positions[i][1]]
            if value_in_cell > 2:
                item_type = value_in_cell - 2  # type of item is value in cell - 2
                temp_rewards[i][item_type] = temp_rewards[i][item_type] + 1
                self.env_map[new_positions[i][0], new_positions[i][1]] = 0

        # section_consumptions, section_agent_types = self._get_stats()
        #
        # self.current_timestep += 1
        #
        # env_termination = self.current_timestep >= self.num_timesteps
        # self.terminations = {agent: env_termination for agent in self.agents}
        # reward_per_section = np.zeros((self.sections, NUM_OBJECTIVES), dtype=np.float32)
        #
        # if env_termination:
        #     if self.reward_scheme == "local":
        #         for i in range(self.sections):
        #             lr_capacity = _local_capacity_reward(self.resource_capacities[i], section_consumptions[i])
        #             lr_mixture = _local_mixture_reward(section_agent_types[i])
        #             reward_per_section[i] = np.array([lr_capacity, lr_mixture])
        #
        #     elif self.reward_scheme == "global":
        #         g_capacity = _global_capacity_reward(self.resource_capacities, section_consumptions)
        #         g_mixture = _global_mixture_reward(section_agent_types)
        #         reward_per_section = np.array([[g_capacity, g_mixture]] * self.sections)
        #
        # Obs: agent type, section id, section capacity, section consumption, % of agents of current type
        observations = {agent: None for agent in self.agents}

        for i, agent in enumerate(self.agents):
            observations[agent] = self._get_obs(i, section_consumptions, section_agent_types)
            rewards[agent] = temp_rewards[i]

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_termination:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, self.truncations, self.terminations, infos
        """

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
        obs = deepcopy(self.env_map)
        for i in range(len(self.agent_positions)):
            if i == agent_id:
                marker = 1
            else:
                marker = 2
            obs[self.agent_positions[i][0], self.agent_positions[i][1]] = marker
        return obs
