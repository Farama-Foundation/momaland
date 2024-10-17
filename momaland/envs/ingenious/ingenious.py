"""Ingenious environment.

|--------------------|--------------------------------------------------------------|
| Actions            | Discrete                                                     |
| Parallel API       | No                                                           |
| Manual Control     | No                                                           |
| Agents             | num_agents=2                                                 |
| Action Shape       | (1,)                                                         |
| Action Values      | Discrete(size depends on board size and rack size: there     |
|                    |  is one integer encoding the placement of each rack tile     |
|                    |  on each board hex in each possible direction.)              |
| Observations       | Observations are dicts with three entries:                   |
|                    |  "board": array with size (2*board_size-1, 2*board_size-1)   |
|                    |  containing values from 0 to num_colors;                     |
|                    |  "racks": for each observable agent, an array of length      |
|                    |  rack_size containing pairs of values from 0 to num_colors;  |
|                    |  "scores": for all agents, their scores in all num_colors    |
|                    |  objectives as values from 0 to max_score.                   |
| Reward Shape       | (num_colors=6,)                                              |

This environment is based on the Ingenious game: https://boardgamegeek.com/boardgame/9674/ingenious

The game's original rules support multiple players collecting scores in multiple colors, which we define as the
objectives of the game: for example (red=5, green=2, blue=9). The goal in the original game is to maximize the
minimum score over all colors (2 in the example above), however we leave the utility wrapper up to the users and only
return the vectorial score on each color dimension (5,2,9).


### Observation Space

The observation is a dictionary which contains an 'observation' element which is the usual RL observation,
and an 'action_mask' which holds the legal moves, described in the Legal Actions Mask section below.

The 'observation' element itself is a dictionary with three entries: 'board' is representing the hexagonal board as
an array of size (2*board_size-1, 2*board_size-1) with integer entries from 0 (empty hex) to num_colors (tiles of
different colors). 'racks' represents for each observable agent - by default only the acting agent, if fully_obs=True
all agents - their tiles rack as an array of size rack_size containing pairs of integers (each pair is a tile) from 0
to num_colors. 'scores' represents for all agents their current scores in all num_colors objectives, as integers from
0 to max_score.


#### Legal Actions Mask

The legal moves available to the current agent are found in the 'action_mask' element of the dictionary observation.
The 'action_mask' is a binary vector where each index of the vector represents whether the represented action is legal
or not; the action encoding is described in the Action Space section below.
The 'action_mask' shows only the current agent's legal moves.


### Action Space

The action space depends on board size and rack size: It contains one integer for each possible placement of any of
the player's rack tiles (rack_size parameter) on any board hex (board_size parameter) in every possible direction.


### Rewards

The agents can collect a separate score in each available color. These scores are the num_colors different reward
dimensions.


### Version History

"""

import functools
import random
from typing_extensions import override

import numpy as np
from gymnasium.logger import warn
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils import EzPickle
from pettingzoo.utils import wrappers

from momaland.envs.ingenious.ingenious_base import ALL_COLORS, IngeniousBase
from momaland.utils.env import MOAECEnv


def env(**kwargs):
    """Returns the wrapped Ingenious environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function

    Returns:
        A fully wrapped AEC env
    """
    env = raw_env(**kwargs)

    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


def raw_env(**kwargs):
    """Env factory function for the Ingenious environment."""
    return Ingenious(**kwargs)


class Ingenious(MOAECEnv, EzPickle):
    """Environment for the Ingenious board game."""

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
            board_size (int): The size of one side of the hexagonal board (between 3 and 10). By default the size is set
             to n+4 where n is the number of agents.
            reward_mode (str): Can be set to "competitive" (individual rewards for all agents), "collaborative" (shared
            rewards for all agents), or "two_teams" (rewards shared within two opposing teams; num_agents needs to be
            even). Default is "competitive".
            fully_obs (bool): Fully observable game mode, i.e. the racks of all players are visible. Default is False.
            render_mode (str): The rendering mode. Default: None
        """
        EzPickle.__init__(
            self,
            num_agents,
            rack_size,
            num_colors,
            board_size,
            reward_mode,
            fully_obs,
            render_mode,
        )
        self.num_colors = num_colors
        self.init_draw = rack_size
        self.max_score = 18  # max score in score board for one certain color.
        assert reward_mode in {
            "competitive",
            "collaborative",
            "two_teams",
        }, "reward_mode has to be one element in {'competitive','collaborative','two_teams'}"
        self.reward_mode = reward_mode
        self.fully_obs = fully_obs

        if self.reward_mode == "two_teams":
            assert num_agents % 2 == 0, "Number of players must be even if reward_mode is two_teams."
            self.max_score = self.max_score * (num_agents / 2)
        elif self.reward_mode == "collaborative":
            self.max_score = self.max_score * num_agents

        if board_size is None:
            self.board_size = {2: 6, 3: 7, 4: 8, 5: 9, 6: 10}.get(num_agents)
        else:
            self.board_size = board_size

        self.game = IngeniousBase(
            num_agents=num_agents,
            rack_size=self.init_draw,
            num_colors=self.num_colors,
            board_size=self.board_size,
            max_score=self.max_score,
        )

        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.game.agent_selector]
        self._cumulative_rewards = {agent: np.zeros(self.num_colors) for agent in self.agents}
        self.refresh_cumulative_reward = True
        self.render_mode = render_mode

        self.observation_spaces = {
            i: Dict(
                {
                    "observation": Dict(
                        {
                            "board": Box(
                                0, len(ALL_COLORS), shape=(2 * self.board_size - 1, 2 * self.board_size - 1), dtype=np.float32
                            ),
                            "racks": (
                                Box(0, self.num_colors, shape=(num_agents, self.init_draw, 2), dtype=np.int32)
                                if self.fully_obs
                                else Box(0, self.num_colors, shape=(self.init_draw, 2), dtype=np.int32)
                            ),
                            "scores": Box(0, self.game.max_score, shape=(num_agents, self.num_colors), dtype=np.int32),
                        }
                    ),
                    "action_mask": Box(low=0, high=1, shape=(len(self.game.masked_action),), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.action_spaces = dict(zip(self.agents, [Discrete(len(self.game.masked_action))] * num_agents))

        # The reward for each move is the difference between the previous and current score.
        self.reward_spaces = dict(zip(self.agents, [Box(0, self.game.max_score, shape=(self.num_colors,))] * num_agents))

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
        and step() can be called without issues."""
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

        # update teammate score (copy current agent's score to teammates)
        if self.reward_mode != "competitive":
            index_current_agent = self.agents.index(current_agent)
            for i in range(0, self.num_agents):
                if self.reward_mode == "two_teams":
                    # in two_team mode, players who are teammates of the current agent get the same reward and score
                    if i != index_current_agent and i % 2 == index_current_agent % 2:
                        agent = self.agents[i]
                        self.game.score[agent] = self.game.score[current_agent]
                        self.rewards[agent] = self.rewards[current_agent]
                elif self.reward_mode == "collaborative":
                    # in collaborative mode, every player gets the same reward and score
                    if i != index_current_agent:
                        agent = self.agents[i]
                        self.game.score[agent] = self.game.score[current_agent]
                        self.rewards[agent] = self.rewards[current_agent]

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
            p_tiles = np.array(self.game.p_tiles[agent], dtype=np.int32)
        tmp = []
        for agent_score in self.game.score.values():
            tmp.append([score for score in agent_score.values()])
        p_score = np.array(tmp, dtype=np.int32)
        observation = {"board": board_vals, "racks": p_tiles, "scores": p_score}
        action_mask = np.array(self.game.return_action_list(), dtype=np.int8)
        return {"observation": observation, "action_mask": action_mask}
