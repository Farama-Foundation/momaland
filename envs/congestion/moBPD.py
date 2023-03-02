import functools
import random
import gymnasium
from gymnasium.spaces import Discrete
import numpy as np

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from resources import Resource

LEFT = -1
RIGHT = 1
STAY = 0
MOVES = ["LEFT", "RIGHT", "STAY"]
NUM_ITERS = 1000


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
    def __init__(self, sections=6, capacity=10, num_agents=100, mode='uniform', render_mode=None):
        self.sections = sections
        # Extend to distinct capacities per section?
        self.resources = [Resource(i, capacity) for i in range(sections)]

        self.episode_num = 0
        self.num_agents = num_agents

        self.render_mode = render_mode
        self.agents = ["agent_" + str(r) for r in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(num_agents))))
        self.types, self.state = self.init_state(mode)

        self.action_spaces = dict(
            zip(self.agents, [gymnasium.spaces.Discrete(len(MOVES))]*num_agents)
        )
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    gymnasium.spaces.Box(
                        low=0,
                        high=num_agents,
                        # Each section is defined by capacity, consumption, mixture
                        # Agent can also observe its type
                        shape=(4, 1),
                        dtype=np.float32,
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
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        for r in self.resources:
            r.print_resource()

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
        self.state = self.init_state()
        observations = {agent: None for agent in self.agents}
        self.episode_num = 0

        if not return_info:
            return observations
        else:
            infos = {agent: {} for agent in self.agents}
            return observations, infos

    def init_state(self, mode):
        if mode == 'uniform':
            typeA = self.num_agents/2
            types = np.concatenate((np.zeros(typeA), np.ones(self.num_agents-typeA)))
            state = [random.randint(0,self.sections-1) for _ in self.agents]
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

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        reward_per_section = []
        groups = []

        actions = [actions[agent] for agent in self.agents]

        # Apply actions and update system state
        for i, agent in enumerate(self.agents):
            act = actions[agent]
            self.state[i] = min(self.sections-1, max(self.state[i] + act, 0))

        # Group agents per section, according to their new states
        for s in range(self.sections):
            groups.append([i for i in range(self.num_agents) if self.state[i] == s])

        # Update resources and get rewards
        for i, group in enumerate(groups):
            self.resources[i].add_load(group, [self.types[index] for index in group])
            reward_per_section.append(self.resources[i].local_reward())

        # Distribute rewards per agent
        rewards = {self.agents[i]: reward_per_section[self.state[i]] for i in range(self.num_agents)}

        terminations = {agent: False for agent in self.agents}

        self.episode_num += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # current observation is the agent's own type and the occupied resource capacity, consumption, mixture
        for i, agent in enumerate(self.agents):
            observations = {agent: [self.types[i]].append(self.resources[self.state[i]].get_obs(self.types[i]))}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos