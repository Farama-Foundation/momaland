"""Multi-objective Ingenious environment for MOMAland.

To Write.
"""

import functools
import random

# from gymnasium.utils import EzPickle
from typing_extensions import override

import numpy as np
from gymnasium.logger import warn
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo.utils import wrappers

from momaland.envs.ingenious.ingenious_base import ALL_COLORS, IngeniousBase
from momaland.utils.env import MOAECEnv


def env(**kwargs):
    """Autowrapper for multi-objective Ingenious game.

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
    """Env factory function for multi-objective Ingenious game."""
    return MOIngenious(**kwargs)


class MOIngenious(MOAECEnv):
    """Environment for the multi-objective Ingenious game."""

    metadata = {"render_modes": ["human"], "name": "moingenious_v0", "is_parallelizable": False}

    def __init__(
        self,
        num_agents: int = 2,
        rack_size: int = 6,
        num_colors: int = 6,
        board_size: int = None,
        reward_mode: str = "competitive",
        fully_obs: bool = False,
        render_mode: bool = None,
    ):
        """Initializes the Ingenious environment.

        Args:
            num_agents (int): The number of agents (between 2 and 6). Default is 2.
            rack_size (int): The number of tiles each player keeps in their rack (between 2 and 6). Default is 6.
            num_colors (int): The number of colors (objectives) in the game (between 2 and 6). Default is 6.
            board_size (int): The size of one side of the hexagonal board (between 3 and 10). By default the size is set to n+4 where n is the number of agents.
            reward_mode (str): Can be set to "competitive" (individual rewards for all agents), "collaborative" (shared rewards for all agents), or "two_teams" (rewards shared within two opposing teams; num_agents needs to be even). Default is "competitive".
            fully_obs (bool): Fully observable game mode, i.e. the racks of all players are visible. Default is False.
            render_mode (str): The rendering mode. Default: None
        """
        self.num_colors = num_colors
        self.init_draw = rack_size
        self.num_players = num_agents
        self.limitation_score = 18  # max score in score board for one certain color.
        assert reward_mode in {
            "competitive",
            "collaborative",
            "two_teams",
        }, "reward_mode has to be one element in {'competitive','collaborative','two_teams'}"
        self.teammate_mode = reward_mode

        if self.teammate_mode == "two_teams":
            assert num_agents % 2 == 0, "Number of players must be even if teammate_mode is two_teams."
            self.limitation_score = self.limitation_score * (num_agents / 2)
        elif self.teammate_mode == "collaborative":
            self.limitation_score = self.limitation_score * num_agents

        self.fully_obs = fully_obs
        if board_size is None:
            self.board_size = {2: 6, 3: 7, 4: 8, 5: 9, 6: 10}.get(self.num_players)
        else:
            self.board_size = board_size

        self.game = IngeniousBase(
            num_agents=self.num_players,
            rack_size=self.init_draw,
            num_colors=self.num_colors,
            board_size=self.board_size,
            max_score=self.limitation_score,
        )

        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        # init list of agent
        self.agents = self.possible_agents[:]

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.game.agent_selector]
        self._cumulative_rewards = {agent: np.zeros(self.num_colors) for agent in self.agents}
        self.refresh_cumulative_reward = True
        self.render_mode = render_mode

        # Observation space is a dict of 2 elements: actions mask and game state (board, agent own tile bag,
        # agent score)
        self.observation_spaces = {
            i: Dict(
                {
                    "observation": Dict(
                        {
                            "board": Box(
                                0, len(ALL_COLORS), shape=(2 * self.board_size - 1, 2 * self.board_size - 1), dtype=np.float32
                            ),
                            "tiles": Box(0, self.num_colors, shape=(self.init_draw,), dtype=np.int32),
                            "scores": Box(0, self.game.limitation_score, shape=(self.num_colors,), dtype=np.int32),
                        }
                    ),
                    "action_mask": Box(low=0, high=1, shape=(len(self.game.masked_action),), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.action_spaces = dict(zip(self.agents, [Discrete(len(self.game.masked_action))] * num_agents))

        # The reward after one move is the difference between the previous and current score.
        self.reward_spaces = dict(
            zip(self.agents, [Box(0, self.game.limitation_score, shape=(self.num_colors,))] * num_agents)
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
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return

    @override
    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(),
        and step() can be called without issues.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.game.reset_game(seed)
        self.agents = self.possible_agents[:]
        obs = {agent: self.observe(agent) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.game.agent_selector]
        self.agent_selection = self.agents[self.game.agent_selector]
        self.rewards = {agent: np.zeros(self.num_colors, dtype="float64") for agent in self.agents}
        self._cumulative_rewards = {agent: np.zeros(self.num_colors, dtype="float64") for agent in self.agents}
        self.refresh_cumulative_reward = True
        return obs, self.infos

    @override
    def step(self, action):
        """Steps in the environment.

        Args:
            action: action of the active agent
        """

        current_agent = self.agent_selection

        if self.terminations[current_agent] or self.truncations[current_agent]:
            return self._was_dead_step(action)
        self.rewards = {agent: np.zeros(self.num_colors, dtype="float64") for agent in self.agents}
        if self.refresh_cumulative_reward:
            self._cumulative_rewards[current_agent] = np.zeros(self.num_colors, dtype="float64")

        # update current agent
        if not self.game.end_flag:
            prev_rewards = np.array(list(self.game.score[current_agent].values()))
            self.game.set_action_index(action)
            current_rewards = np.array(list(self.game.score[current_agent].values()))
            self.rewards[current_agent] = current_rewards - prev_rewards

        if self.game.end_flag:
            self.terminations = {agent: True for agent in self.agents}

        # update teammate score(copy current agent score to the teammate)
        if self.teammate_mode != "competitive":
            index_current_agent = self.agents.index(current_agent)
            for i in range(0, self.num_players):
                if self.teammate_mode == "two_teams":
                    # two team mode, players who is teammates of the current agent has the same reward and score
                    if i != index_current_agent and i % 2 == index_current_agent % 2:
                        agent = self.agents[i]
                        self.game.score[agent] = self.game.score[current_agent]
                        self.rewards[agent] = self.rewards[current_agent]
                elif self.teammate_mode == "collaborative":
                    # collabarotive mode, every player has the same reward and score
                    if i != index_current_agent:
                        agent = self.agents[i]
                        self.game.score[agent] = self.game.score[current_agent]
                        self.rewards[agent] = self.rewards[current_agent]

        # update accumulate_rewards
        self._accumulate_rewards()

        # update to next agent
        self.agent_selection = self.agents[self.game.agent_selector]

        if self.agent_selection != current_agent:
            self.refresh_cumulative_reward = True
        else:
            self.refresh_cumulative_reward = False

    @override
    def observe(self, agent):
        board_vals = np.array(self.game.board_array, dtype=np.float32)
        if self.fully_obs:
            p_tiles = np.array([item for item in self.game.p_tiles.values()], dtype=np.int32)
        else:
            # print(self.game.p_tiles[agent])
            p_tiles = np.array(self.game.p_tiles[agent], dtype=np.int32)
            # p_score = np.array(list(self.game.score[agent].values()), dtype=np.int32)
        # show all score board
        tmp = []
        for agent_score in self.game.score.values():
            tmp.append([score for score in agent_score.values()])
        p_score = np.array(tmp, dtype=np.int32)

        observation = {"board": board_vals, "tiles": p_tiles, "scores": p_score}
        action_mask = np.array(self.game.return_action_list(), dtype=np.int8)
        return {"observation": observation, "action_mask": action_mask}
