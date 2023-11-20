"""Breakthrough.

|--------------------|--------------------------------------------------|
| Actions            | Discrete                                         |
| Parallel API       | No                                               |
| Manual Control     | No                                               |
| Agents             | 2                                                |
| Action Shape       | (1,)                                             |
| Action Values      | Discrete(board_width=8 * board_height=8 * 3)     |
| Observation Shape  | (board_height=8, board_width=8, 2)               |
| Observation Values | [0,1]                                            |
| Reward Shape       | (3,)                                             |

Breakthrough is a 2-player turn based game, where players must try to reach the opponent's home row with any of their
pieces. The first player to move a piece there wins. Players move alternatingly, and each piece can move one square
straight forward or diagonally forward. Opponent pieces can also be captured, but only by moving diagonally forward,
not straight.


### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described
below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

The main observation space is 2 planes of a board_heightxboard_width grid. Each plane represents a specific agent's
tokens, and each location in the grid represents the placement of the corresponding agent's token. 1 indicates that
the agent has a token placed in that cell, and 0 indicates they do not have a token in that cell. A 0 means that
either the cell is empty, or the other agent has a token in that cell.


#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation.
The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not.
The `action_mask` will be all zeros for any agent except the one whose turn it is. Taking an illegal move ends the
game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents. #TODO this isn't happening anymore because of missing TerminateIllegalWrapper


### Action Space

The action space is the set of integers from 0 to board_width*board_height*3 (exclusive). If a piece at coordinates (
x,y) is moved, this is encoded as the integer x*y+z where z == 0 for left diagonal, 1 for straight, and 2 for right
diagonal move.


### Rewards

Dimension 0: If an agent moves one of their pieces to the opponent's home row, they will be rewarded 1 point. At the
same time, the opponent agent will be awarded -1 point. There are no draws in Breakthrough. Dimension 1: If an agent
wins, they get a reward of 1-(move_count/max_moves) to incentivize faster wins. The losing opponent gets the negated
reward. In case of a draw, both agents get 0. Dimension 2: (optional) The number of opponent pieces (divided by the
max number of pieces) an agent has captured. Dimension 3: (optional) The negative number of pieces (divided by the
max number of pieces) an agent has lost to the opponent.


### Version History

"""
from __future__ import annotations

from typing_extensions import override

import numpy as np
from gymnasium import spaces
from gymnasium.logger import warn
from pettingzoo.utils import agent_selector, wrappers

from momaland.utils import mo_aec_to_parallel
from momaland.utils.env import MOAECEnv


OFF_BOARD = -1
ANGLES = ["LEFT", "STRAIGHT", "RIGHT"]


def env(**kwargs):
    """Returns the wrapped MOBreakthrough environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function

    Returns:
        A fully wrapped AEC env
    """
    env = raw_env(**kwargs)
    # This wrapper terminates the game with the current player losing in case of illegal values.
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    # Asserts if the action given to step is outside the action space.
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Checks if function calls or attribute access are in a disallowed order.
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


def parallel_env(**kwargs):
    """Returns the wrapped MOBreakthrough environment in `parallel` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A fully wrapped parallel env.
    """
    env = raw_env(**kwargs)
    env = mo_aec_to_parallel(env)
    return env


def raw_env(**kwargs):
    """Returns the MOBreakthrough environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to create the `MOBreakthrough` environment.

    Returns:
        A raw env.
    """
    return MOBreakthrough(**kwargs)


def human_print(array):
    """Prints the board in a human-readable format."""
    print(np.rot90(array))


class MOBreakthrough(MOAECEnv):
    """Breakthrough environment with multiple objectives."""

    metadata = {
        "render_modes": ["human"],
        "name": "mobreakthrough_v0",
        "is_parallelizable": True,  # TODO ?
    }

    def __init__(self, board_width: int = 8, board_height: int = 8, num_objectives=4, render_mode=None):
        """Initialize the MOBreakthrough environment.

        Args:
            board_width: The width of the board.
            board_height: The height of the board.
            num_objectives: The number of objectives.
            render_mode: The render mode.
        """
        if not (3 <= board_width <= 20):
            raise ValueError("Config parameter board_width must be between 3 and 20.")

        elif not (5 <= board_height <= 20):
            raise ValueError("Config parameter board_height must be between 5 and 20.")

        elif not (2 <= num_objectives <= 4):
            raise ValueError("Config parameter num_objectives must be between 2 and 4.")

        self.board_width = board_width
        self.board_height = board_height
        self.num_objectives = num_objectives
        self.board_size = board_height * board_width
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.move_count = 0
        self.max_turns = 4 * board_width * (board_height - 3) + 1
        self.max_move = board_height * board_width * 3
        self._cumulative_rewards = {i: np.zeros(self.num_objectives) for i in self.agents}

        self._initialize_board(board_height, board_width)
        self.legal_moves = self._legal_moves()
        self.render_mode = render_mode

        self.action_spaces = {i: spaces.Discrete(self.max_move) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(board_height, board_width, len(self.agents)), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(self.max_move,), dtype=np.int8),
                }
            )
            for i in self.agents
        }
        self.reward_spaces = dict(
            zip(self.agents, [spaces.Box(low=-1, high=1, shape=(self.num_objectives,))] * len(self.agents))
        )

    @override
    def render(self):
        """Renders the environment.

        In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return

    # Key
    # ----
    # blank space = 0
    # agent 0 = 1
    # agent 1 = 2
    # An observation is list of lists, where each list represents a row
    # E.g.
    # array([[0, 1, 1, 2, 0, 1, 0],
    #        [1, 0, 1, 2, 2, 2, 1],
    #        [0, 1, 0, 0, 1, 2, 1],
    #        [1, 0, 2, 0, 1, 1, 0],
    #        [2, 0, 0, 0, 1, 1, 0],
    #        [1, 1, 2, 1, 0, 1, 0]], dtype=int8)

    @override
    def observe(self, agent):
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2
        cur_p_board = np.equal(self.board, cur_player + 1)
        opp_p_board = np.equal(self.board, opp_player + 1)
        observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
        actions = self.legal_moves if agent == self.agent_selection else []
        action_mask = np.zeros(self.max_move, "int8")
        for i in actions:
            action_mask[i] = 1
        return {"observation": observation, "action_mask": action_mask}

    @override
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @override
    def action_space(self, agent):
        return self.action_spaces[agent]

    @override
    def reward_space(self, agent):
        return self.reward_spaces[agent]

    def _legal_moves(self):
        """Returns a list of legal moves for the current player."""
        legal_moves = set()
        cur_player = self.possible_agents.index(self.agent_selection)
        opp_player = (cur_player + 1) % 2
        cur_piece = cur_player + 1
        opp_piece = opp_player + 1
        move_direction = 1 if cur_piece == 1 else -1
        for index, value in np.ndenumerate(self.board):
            x, y = index
            if value == cur_piece:
                if self._get_square(x, y + move_direction) == 0:  # move straight ahead possible
                    legal_moves.add(self._move_to_int(x, y, "STRAIGHT"))
                    # print("legal: straight from ", x, ",", y)
                    # print("to move: ", self._move_to_int(x, y, "STRAIGHT"))
                    # print("back to coordinates: ", self._int_to_move(self._move_to_int(x, y, "STRAIGHT")))
                if x < self.board_width - 1:
                    if (
                        self._get_square(x + 1, y + move_direction) == 0
                        or self._get_square(x + 1, y + move_direction) == opp_piece
                    ):  # move diagonally to the right possible
                        legal_moves.add(self._move_to_int(x, y, "RIGHT"))
                if x > 0:
                    if (
                        self._get_square(x - 1, y + move_direction) == 0
                        or self._get_square(x + 1, y + move_direction) == opp_piece
                    ):  # move diagonally to the left possible
                        legal_moves.add(self._move_to_int(x, y, "LEFT"))
        return legal_moves

    def _get_square(self, x, y):
        """Returns the piece at the given coordinates, provided the coordinates are legal."""
        if x < 0 or x > self.board_width - 1 or y < 0 or y > self.board_height - 1:
            return OFF_BOARD
        return self.board[x][y]

    @override
    def step(self, action):
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)  # TODO is this needed?

        # assert valid move
        assert action in self.legal_moves, "played illegal move."

        x, y, direction = self._int_to_move(action)
        agent = self.agent_selection
        agent_index = self.possible_agents.index(agent)
        next_agent = self.possible_agents[(agent_index + 1) % 2]
        agent_piece = agent_index + 1
        move_direction = 1 if agent_piece == 1 else -1
        self.board[x][y] = 0
        capture = False
        if self.board[x + ANGLES.index(direction) - 1][y + move_direction] != 0:
            capture = True
        self.board[x + ANGLES.index(direction) - 1][y + move_direction] = agent_piece
        self.move_count += 1
        winner = self.check_for_winner()

        # self.rewards = {i: np.array([0] * self.num_objectives, dtype=np.float32) for i in self.agents}
        # self._cumulative_rewards[agent] = np.array([0] * self.num_objectives, dtype=np.float32)
        if capture:
            if self.num_objectives > 2:
                self.rewards[agent][2] = 1 / (self.board_width * 2)
            if self.num_objectives > 3:
                self.rewards[next_agent][3] = -1 / (self.board_width * 2)
        # check if there is a winner
        if winner:
            self.rewards[agent][0] = 1
            self.rewards[next_agent][0] = -1
            self.rewards[agent][1] = 1 - (self.move_count / self.max_turns)
            self.rewards[next_agent][1] = -(1 - (self.move_count / self.max_turns))
            self.terminations = {i: True for i in self.agents}

        self._cumulative_rewards[agent] = np.array([0] * self.num_objectives, dtype=np.float32)
        self._accumulate_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        self.legal_moves = self._legal_moves()

        if self.render_mode == "human":
            self.render()

    @override
    def reset(self, seed=None, options=None):
        # reset environment
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {i: np.zeros(self.num_objectives) for i in self.agents}
        self._cumulative_rewards = {i: np.zeros(self.num_objectives) for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.move_count = 0
        self._initialize_board(self.board_height, self.board_width)
        self.legal_moves = self._legal_moves()

    @override
    def close(self):
        """Close should release any graphical displays, subprocesses, network connections or any other environment
        data which should not be kept around after the user is no longer using the environment."""
        pass

    def check_for_winner(self):
        """Check if there is a winner."""
        cur_player = self.possible_agents.index(self.agent_selection)
        opp_player = (cur_player + 1) % 2
        cur_piece = cur_player + 1
        opp_piece = opp_player + 1
        if not (opp_piece in self.board):
            return True
        if cur_piece == 1:
            home_row = self.board_height - 1
        else:
            home_row = 0
        if cur_piece in self.board[:, home_row]:
            return True
        return False

    def _initialize_board(self, board_height, board_width):
        """Initialize the board."""
        self.board = np.zeros((board_width, board_height))
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        piece = self.agents.index(self.agent_selection) + 1
        next_agent_index = piece % 2
        opp_piece = next_agent_index + 1
        self.board[:, :2] = piece
        self.board[:, -2:] = opp_piece

    def _move_to_int(self, x, y, direction):
        """Convert move coordinates and direction to integer move encoding."""
        return x * 3 * self.board_height + y * 3 + ANGLES.index(direction)

    def _int_to_move(self, move):
        """Convert integer move encoding to move coordinates and direction."""
        return (move // 3) // self.board_height, (move // 3) % self.board_height, ANGLES[move % 3]
