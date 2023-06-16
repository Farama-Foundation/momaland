import functools
import random
from gymnasium.spaces import Discrete, Box
from gymnasium.logger import warn
#from gymnasium.utils import EzPickle
import numpy as np

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

LEFT = -1
RIGHT = 1
STAY = 0
MOVES = ["LEFT", "RIGHT", "STAY"]
NUM_OBJECTIVES = 2


def env(**kwargs):
    env = raw_env(**kwargs)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "mobeach_v0"}
    """
    The init method takes in environment arguments and should define the following attributes:
    - possible_agents
    - action_spaces
    - observation_spaces
    These attributes should not be changed after initialization.
    """
    def __init__(self, sections=6,
                 capacity=10,
                 num_agents=100,
                 init_position='random',
                 init_type='random',
                 num_types=2,
                 num_timesteps=10,
                 render_mode=None):
        self.sections = sections
        #TODO Extend to distinct capacities per section?
        self.resource_capacities = [capacity for _ in range(sections)]
        self.num_types = num_types
        self.num_timesteps = num_timesteps
        self.episode_num = 0
        self.init_position = init_position
        self.init_type = init_type
        self.render_mode = render_mode
        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.types, self.state = self.init_state()

        self.action_spaces = dict(
            zip(self.agents, [Discrete(len(MOVES))]*num_agents)
        )
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

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasiuspspom.farama.org/api/spaces/
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            warn(
                "You are calling render method without specifying any render mode."
            )
            return

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.types, self.state = self.init_state()
        observations = {agent: None for agent in self.agents}
        self.episode_num = 0

        if not return_info:
            return observations
        else:
            infos = {agent: {} for agent in self.agents}
            return observations, infos

    def init_state(self):
        if self.init_type == 'random':
            types = [random.randint(0, self.num_types-1) for _ in self.agents]
        if self.init_position == 'random':
            state = [random.randint(0, self.sections-1) for _ in self.agents]
        return types, state

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
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
            self.state[i] = min(self.sections-1, max(self.state[i] + act, 0))

        section_consumptions = np.zeros(self.sections)
        section_agent_types = np.zeros((self.num_types, self.sections))

        for i in range(len(self.agents)):
            section_consumptions[self.state[i]] += 1
            section_agent_types[self.state[i]][self.types[i]] += 1

        self.episode_num += 1

        env_termination = self.episode_num >= self. num_timesteps
        reward_per_section = np.zeros((self.sections, NUM_OBJECTIVES))

        #TODO split in separate functions
        if env_termination:
            for i in range(self.sections):
                lr_capacity = section_consumptions[i] * np.exp(-section_consumptions[i] / self.resource_capacities[i])
                t = section_agent_types[i]
                if sum(t) > 0:
                    lr_mixture = min(t) / sum(t)
                else:
                    lr_mixture = 0
                reward_per_section[i] = np.array([lr_capacity, lr_mixture])

        # Obs: agent type, section id, section capacity, section consumption, % of agents of current type
        observations = {agent: None for agent in self.agents}
        # Note that agents only receive the reward after the last timestep
        rewards = {self.agents[i]: [0, 0] for _ in range(self.num_agents)}

        for i, agent in enumerate(self.agents):
            total_same_type = section_agent_types[self.state[i]][self.types[i]]
            t = total_same_type/section_consumptions[self.state[i]]
            obs = [self.types[i],
                   self.state[i],
                   self.resource_capacities[self.state[i]],
                   section_consumptions[self.state[i]],
                   t]
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