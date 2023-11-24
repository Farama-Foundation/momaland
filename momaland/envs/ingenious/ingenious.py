"""Multi-objective Ingenious environment for MOMAland.

This environment is based on the Ingenious game: https://boardgamegeek.com/boardgame/9674/ingenious
Every color is a different objective. The goal in the original game is to maximize the minimum score over all colors,
however we leave the utility wrapper up to the users and only return the vectorial score on each color dimension.
"""

import functools
import random

# from gymnasium.utils import EzPickle
from typing_extensions import override

import numpy as np
from gymnasium.logger import warn
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo.utils import wrappers

from momaland.envs.ingenious.ingenious_base import IngeniousBase
from momaland.utils.conversions import mo_aec_to_parallel
from momaland.utils.env import MOAECEnv


def parallel_env(**kwargs):
    """Parallel env factory function for the ingenious environment."""
    env = raw_env(**kwargs)
    env = mo_aec_to_parallel(env)
    return env(**kwargs)


def env(**kwargs):
    """Autowrapper for the item gathering problem.

    Args:
        **kwargs: keyword args to forward to the parallel_env function

    Returns:
        A fully wrapped env
    """
    env = raw_env(**kwargs)

    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


def raw_env(**kwargs):
    """Env factory function for the item gathering problem."""
    return MOIngenious(**kwargs)


class MOIngenious(MOAECEnv):
    """Environment for the multi-objective Ingenious game."""

    metadata = {"render_modes": ["human"], "name": "moingenious_v0"}

    def __init__(self, num_players=2, init_draw=6, num_colors=4, board_size=8):
        """Initializes the ingenious game.

        Args:
            num_players (int): The number of players in the environment. Default: 2
            init_draw (int): The number of tiles each player draws at the beginning of the game. Default: 6
            num_colors (int): The number of colors in the game. Default: 4
            board_size (int): The size of the board. Default: 8
        """
        self.board_size = board_size
        self.num_colors = num_colors
        self.init_draw = init_draw
        self.num_players = num_players

        self.game = IngeniousBase(num_players=num_players, init_draw=init_draw, num_colors=num_colors, board_size=board_size)

        self.possible_agents = ["agent_" + str(r) for r in range(num_players)]
        # init list of agent
        self.agents = self.possible_agents[:]

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.game.agent_selector]
        self._cumulative_rewards = {agent: np.zeros(self.num_colors) for agent in self.agents}

        # Observation space is a dict of 2 elements: actions mask and game state (board, agent own tile bag,
        # agent score)
        self.observation_space = {
            i: Dict(
                {
                    "observation": Dict(
                        {
                            "board": Box(0, self.num_colors, shape=(self.board_size, self.board_size)),
                            "tiles": Box(0, self.num_colors, shape=(self.init_draw, 2)),
                            "scores": Box(0, self.game.limitation_score, shape=(self.num_colors,)),
                        }
                    ),
                    "action_mask": Box(low=0, high=1, shape=(len(self.game.masked_action),), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.action_spaces = dict(zip(self.agents, [Discrete(len(self.game.masked_action))] * num_players))

        self.reward_spaces = dict(
            zip(self.agents, [Box(0, self.game.limitation_score, shape=(self.num_colors,))] * num_players)
        )

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
        # self.game.render_game()
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return

    @override
    def close(self):
        """Close should release any graphical displays, subprocesses, network connections or any other
        environment data which should not be kept around after the user is no longer using the environment."""
        pass

    @override
    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(),
        and step() can be called without issues.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.game.reset_game()

    @override
    def step(self, action):
        """Steps in the environment.

        Args:
            action: action of the active agent
        """
        if not self.game.end_flag and self.game.return_action_list()[action] == 1:
            self.game.set_action_index(action)
            self.agent_selection = self.agents[self.game.agent_selector]

        if self.game.end_flag:
            self.terminations = {agent: True for agent in self.agents}
            self.truncations = {agent: True for agent in self.agents}
            self.rewards = {agent: np.array(list(self.game.score[agent].values())) for agent in self.agents}
        else:
            self.rewards = {agent: np.zeros(self.num_colors, dtype="float64") for agent in self.agents}
        # print('before accumulate',self.game.end_flag, self.rewards)
        # self._accumulate_rewards()
        # print('after accumulate')

    @override
    def last(self, observe=True):
        agent = self.agents[self.game.agent_selector]
        assert agent

        observation = self.observe(agent) if observe else None
        return (observation, self.rewards, self.terminations, self.truncations, self.infos)

    @override
    def observe(self, agent):
        board_vals = self.game.board_array
        p_tiles = np.array(self.game.p_tiles[agent])
        p_score = np.array(list(self.game.score[agent].values()))
        # print(p_score)
        # p_index = self.agents[self.game.agent_selector]

        observation = {"board": board_vals, "tiles": p_tiles, "scores": p_score}
        action_mask = self.game.return_action_list()

        # print(observation)
        return {"observation": observation, "action_mask": action_mask}
