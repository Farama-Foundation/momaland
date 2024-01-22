"""Base class for Ingenious environment.

This class is not meant to be instantiated directly. This class supports the MOIngenious environment and provides the
board and rules.
"""

import collections
import itertools
import random

import numpy as np


# red 12-pointed star
# green circle
# blue 6-pointed star
# orange hexagon
# yellow 24-pointed star
# purple ring
RED = 1
GREEN = 2
BLUE = 3
ORANGE = 4
YELLOW = 5
PURPLE = 6
ALL_COLORS = [RED, GREEN, BLUE, ORANGE, YELLOW, PURPLE]
COLOR_NAMES = ["red", "green", "blue", "orange", "yellow", "purple"]

NUM_TILES = 120

# Point = collections.namedtuple("Point", ["x", "y"])

Hex = collections.namedtuple("Hex", ["q", "r", "s"])


def hex_coord(q, r, s):
    """Create a cube-based coordinates."""
    assert not (round(q + r + s) != 0), "q + r + s must be 0"
    return Hex(q, r, s)


def hex_add(a, b):
    """Add two cube-based coordinates."""
    return hex_coord(a.q + b.q, a.r + b.r, a.s + b.s)


def hex_subtract(a, b):
    """Subtract two cube-based coordinates."""
    return hex_coord(a.q - b.q, a.r - b.r, a.s - b.s)


def hex_scale(a, k):
    """Scale a cube-based coordinate."""
    return hex_coord(a.q * k, a.r * k, a.s * k)


def Hex2ArrayLocation(hx, length):
    """Convert cube-based coordinates to 2D-based coordinates."""
    return hx.q + length - 1, hx.r + length - 1


def ArrayLocation2Hex(x, y, length):
    """Convert 2D-based coordinates to cube-based coordinates."""
    q = x + 1 - length
    r = y + 1 - length
    s = -q - r
    return q, r, s


hex_directions = [
    hex_coord(1, 0, -1),
    hex_coord(1, -1, 0),
    hex_coord(0, -1, 1),
    hex_coord(-1, 0, 1),
    hex_coord(-1, 1, 0),
    hex_coord(0, 1, -1),
]


def hex_direction(direction):
    """Return the directions of a hexagon."""
    return hex_directions[direction]


def hex_neighbor(hex, direction):
    """Return the neighbors of a hexagon."""
    return hex_add(hex, hex_direction(direction))


def generate_board(board_size):
    """Generate a hexagonal board."""
    N = board_size - 1
    s = set()
    for q in range(-N, +N + 1):
        r1 = max(-N, -q - N)
        r2 = min(N, -q + N)
        for r in range(r1, r2 + 1):
            location = hex_coord(q, r, -q - r)
            s.add(location)
    return s


class IngeniousBase:
    """Base class for Ingenious environment."""

    def __init__(self, num_players=2, init_draw=6, num_colors=6, board_size=8, limitation_score=18):
        """Initialize the Ingenious environment.

        Args:
            num_players (int): Number of players in the game.
            init_draw (int): Number of tiles to draw at the beginning of the game.
            num_colors (int): Number of colors in the game.
            board_size (int): Size of the board.
            limitation_score(int): Limitation to refresh the score board for any color. Default: 20
        """
        assert 2 <= num_players <= 5, "Number of players must be between 2 and 5."
        assert 2 <= num_colors <= 6, "Number of colors must be between 2 and 6."
        assert 2 <= init_draw <= 6, "Number of tiles in hand must be between 2 and 6."
        assert 3 <= board_size <= 10, "Board size must be between 3 and 10."

        self.board_size = board_size
        self.num_player = num_players
        # self.agents = [r for r in range(num_players)]
        self.agents = [f"agent_{i}" for i in range(self.num_player)]
        self.agent_selector = 0
        self.limitation_score = limitation_score
        self.colors = num_colors
        self.corner_color = ALL_COLORS
        self.init_draw = init_draw
        self.board_array = np.zeros([2 * self.board_size - 1, 2 * self.board_size - 1])
        self.board_hex = generate_board(self.board_size)  # original full board
        self.action_map = {}
        self.action_index_map = {}
        self.action_size = 0
        self.masked_action = []
        self.legal_move = set()
        self.score = {agent: {ALL_COLORS[i]: 0 for i in range(0, self.colors)} for agent in self.agents}

        self.tiles_bag = {}
        self.p_tiles = {agent: [] for agent in self.agents}
        self.first_round = True
        self.first_round_pos = set()
        self.end_flag = False
        self.random = random.Random()

        for loc in self.board_hex:
            for direct in range(0, len(hex_directions)):
                neighbour = hex_neighbor(loc, direct)
                if neighbour not in self.board_hex:
                    continue
                for i in range(0, self.init_draw):
                    if (loc, neighbour, i) not in self.action_map:
                        self.action_map[(loc, neighbour, i)] = self.action_size
                        self.action_index_map[self.action_size] = (loc, neighbour, i)
                        self.legal_move.add(self.action_size)
                        self.action_size += 1
        self.masked_action = np.ones(self.action_size, "int8")
        # self.reset_game()

    def reset_game(self, seed=None):
        """Reset the board, racks, score, and tiles bag."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.random.seed(seed)
            print("seed", seed)
        self.end_flag = False
        self.first_round = True
        self.first_round_pos.clear()
        self.board_array = np.zeros([2 * self.board_size - 1, 2 * self.board_size - 1])
        # generate hex board
        self.board_hex = generate_board(self.board_size)
        # generate action space
        self.action_map.clear()
        self.action_index_map.clear()
        self.action_size = 0
        for loc in self.board_hex:
            for direct in range(0, len(hex_directions)):
                neighbour = hex_neighbor(loc, direct)
                if neighbour not in self.board_hex:
                    continue
                for i in range(0, self.init_draw):
                    if (loc, neighbour, i) not in self.action_map:
                        self.action_map[(loc, neighbour, i)] = self.action_size
                        self.action_index_map[self.action_size] = (loc, neighbour, i)
                        self.legal_move.add(self.action_size)
                        self.action_size += 1
        self.masked_action = np.ones(self.action_size, "int8")

        # generate corner symbol
        self.initial_corner()
        # generate and shuffle public tiles bag
        self.tiles_bag_reset()
        # initial tile draw for each agent
        self.p_tiles = {a: self.draw_tiles_fill() for a in self.agents}
        self.agent_selector = 0
        self.score = {agent: {ALL_COLORS[i]: 0 for i in range(0, self.colors)} for agent in self.agents}

    def draw_tiles_fill(self):
        """Draw tiles for single player with amount(self.init_draw) of tiles."""
        return [self.tiles_bag.pop(self.random.randrange(len(self.tiles_bag))) for _ in range(self.init_draw)]

    def get_tile(self, a):
        """Draw tiles for a specific player."""
        while len(self.p_tiles[a]) < self.init_draw:
            self.p_tiles[a].append(self.tiles_bag.pop(self.random.randrange(len(self.tiles_bag))))
        return

    def initial_corner(self):
        """Initialise the corner of the board with the 6 colors."""
        for i in range(0, 6):
            a = hex_scale(hex_directions[i], self.board_size - 1)
            x, y = Hex2ArrayLocation(a, self.board_size)
            self.board_array[x, y] = self.corner_color[i]
            self.exclude_action(a)

            # In first round, each player has to put the tile next to the the corners position. Therefore, we use self.first_round_pos to maintain the first round position.
            for k in range(0, 6):
                hx1 = hex_neighbor(a, k)
                for j in range(0, 6):
                    hx2 = hex_neighbor(hx1, j)
                    # print(hx1, hx2, "2132")
                    if (hx2 not in self.board_hex) or (hx1 not in self.board_hex) or (hx2 == a):
                        continue
                    # print(hx1, hx2, "21320000")
                    for card in range(0, self.init_draw):
                        c1 = self.action_map[(hx1, hx2, card)]
                        c2 = self.action_map[(hx2, hx1, card)]
                        self.first_round_pos.add(c1)
                        self.first_round_pos.add(c2)

    def tiles_bag_reset(self):
        """Generate and shuffle the tiles bag."""
        # Create a list of tuples for combinations of two different colors
        diff_color_combinations = list(itertools.combinations(ALL_COLORS[: self.colors], 2))
        # Create a list of tuples for combinations of the same integer (color)
        same_color_combinations = [(color, color) for color in ALL_COLORS[: self.colors]]
        # Create the tiles bag
        if self.colors == len(ALL_COLORS):
            # when color type is 6, tiles bag follow the original game setting : six tiles for each two-colour combination (e.g. red/orange) and five for each double (red/red)
            self.tiles_bag = (diff_color_combinations * 6) + (same_color_combinations * 5)
        else:
            # when color type is not 6( like 1-5), the number of combinations could be divided by NUM_TILES(120)
            self.tiles_bag = int(NUM_TILES / len(diff_color_combinations + same_color_combinations)) * (
                diff_color_combinations + same_color_combinations
            )
        # Shuffle the tiles bag
        self.random.shuffle(self.tiles_bag)

    def set_action_index(self, index):
        """Apply the corresponding action for the given index on the board."""
        """If selected actions is not a legal move, return False"""
        assert self.masked_action[index] == 1, "Illegal move, choose a valid action."
        if self.first_round:
            assert index in self.first_round_pos, "illegal move, in the first round tiles can only be placed next to corners."
        """Hex Coordinate: h1,h2 ;  Tile to play: card"""
        h1, h2, card = self.action_index_map[index]
        agent_i = self.agent_selector
        agent = self.agents[agent_i]
        # if card >= len(self.p_tiles[agent]):
        #    assert "illegal move: choosing tile out of hand(happening after ingenious)"
        #    return False
        assert card < len(self.p_tiles[agent]), "illegal move: choosing tile out of hand(happening after ingenious)"
        """Extract the certain tile (color1 , color2) as (c1,c2)"""
        c1, c2 = self.p_tiles[agent][card]
        # Translate Hex Coordinate to Offset Coordinate(x,y)
        x1, y1 = Hex2ArrayLocation(h1, self.board_size)
        x2, y2 = Hex2ArrayLocation(h2, self.board_size)
        flag = False
        for item in self.p_tiles[agent]:
            # print(item)
            if (c1, c2) == item:
                self.p_tiles[agent].remove(item)
                flag = True
                break
            if (c2, c1) == item:
                self.p_tiles[agent].remove(item)
                flag = True
                break
        # if not flag:
        #    assert "illegal move: set the tile to the coordinate unsuccessfully"
        #    return False
        assert flag, "illegal move: set the tile to the coordinate unsuccessfully"
        """Update the mask_action list after the action"""
        self.legal_move.remove(index)
        self.board_array[x1][y1] = c1
        self.board_array[x2][y2] = c2
        self.exclude_action(h1)
        self.exclude_action(h2)
        skip_flag = False
        """Update score through checking 5 neighboring directions for h1 and h2 independently"""
        self.score[agent][c1] += self.calculate_score_for_piece(h1, h2, c1)
        self.score[agent][c2] += self.calculate_score_for_piece(h2, h1, c2)
        if self.score[agent][c1] > self.limitation_score:
            skip_flag = True
            self.score[agent][c1] = 0
        if self.score[agent][c2] > self.limitation_score:
            skip_flag = True
            self.score[agent][c2] = 0

        """End game if no more legal actions."""
        if len(self.legal_move) == 0:
            self.end_flag = True
            return True

        """All tiles in hand has been played"""
        if len(self.p_tiles[agent]) == 0:
            self.end_flag = True  # The player should win instantly if he plays out all the tiles in hand.
            return True

        """Ingenious Situation"""
        if not skip_flag:
            self.get_tile(agent)
            """Swapping your Tiles if tiles in hand has no color with the lowest score"""
            self.refresh_hand(agent)
            self.next_turn()

        # return True

    def calculate_score_for_piece(self, start_hex, other_hex, color):
        """Calculate the scores after placing the tile."""
        point = 0
        for i in range(0, 6):
            neighbor_hex = hex_neighbor(start_hex, i)
            if neighbor_hex == other_hex:
                continue
            while neighbor_hex in self.board_hex:
                x, y = Hex2ArrayLocation(neighbor_hex, self.board_size)
                if self.board_array[x][y] == color:
                    point += 1
                else:
                    break
                neighbor_hex = hex_neighbor(neighbor_hex, i)
        return point

    def exclude_action(self, hx):
        """Exclude the actions that are not legal moves."""
        for i in range(0, 6):
            hx2 = hex_neighbor(hx, i)
            if hx2 not in self.board_hex:
                continue
            for card in range(0, self.init_draw):
                x = self.action_map[(hx, hx2, card)]
                self.masked_action[x] = 0
                if x in self.legal_move:
                    self.legal_move.remove(x)
                y = self.action_map[(hx2, hx, card)]
                self.masked_action[y] = 0
                if y in self.legal_move:
                    self.legal_move.remove(y)

    def next_turn(self):
        """Move to the next turn."""
        self.agent_selector = (self.agent_selector + 1) % self.num_player
        if self.agent_selector == 0 and self.first_round:
            self.first_round = False
        return self.agent_selector

    def refresh_hand(self, player):
        """Additional rule to refresh hand-held tiles."""
        """find the color for which the player has the lowest score"""
        minval = min(self.score[player].values())
        flag_lowest_score = False
        for item in self.p_tiles[player]:
            for col in item:
                if self.score[player][col] == minval:
                    flag_lowest_score = True
            if flag_lowest_score:
                break
        if not flag_lowest_score:
            """no lowest score color"""
            # save current unused tiles to add them back to the tiles bag
            back_up = self.p_tiles[player].copy()
            # clear the player's tiles
            self.p_tiles[player].clear()
            # draw new tiles
            self.get_tile(player)
            # add unused tiles back to the tiles bag
            self.tiles_bag.append(back_up)

    def return_action_list(self):
        """Return the legal action list."""
        if self.first_round:
            return [
                1 if i in self.first_round_pos and self.masked_action[i] == 1 else 0 for i in range(len(self.masked_action))
            ]
        return self.masked_action

    def log(self):
        """Print the current status of the game."""
        print({"board_size": self.board_size, "num_players": self.num_player})
        print("selector", self.agent_selector)
        print(self.board_array)
        print(self.score)
        print(self.p_tiles)
