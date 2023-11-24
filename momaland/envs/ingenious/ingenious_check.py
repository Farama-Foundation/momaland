"""Temporary check file for Ingenious environment."""

import random

from ingenious import MOIngenious
from ingenious_base import Hex2ArrayLocation


def train(ig_env):
    """Train a random agent on the item gathering domain."""
    done = False
    while not done:
        ag = ig_env.agent_selection
        print("Agent: ", ag)
        obs = ig_env.observe(ag)
        masked_act_list = obs["action_mask"]
        action = random_index_of_one(masked_act_list)
        # print("Observation: ",obs)
        # print("Action: ", action)
        ig_env.step(action)
        observation, reward, truncation, termination, _ = ig_env.last()
        print("Observations: ", observation)
        print("Rewards: ", reward[ag])
        print("Truncation: ", truncation[ag])
        print("Termination: ", termination[ag])
        done = truncation[ag] or termination[ag]


def random_index_of_one(lst):
    """Get indices where the value is 1."""
    # Get indices where the value is 1
    one_indices = [i for i, value in enumerate(lst) if value == 1]
    # Check if there is at least one '1' in the list
    if one_indices:
        # Randomly choose an index where the value is 1
        random_index = random.choice(one_indices)
        return random_index
    else:
        # If there are no '1' values in the list, return an appropriate message or handle it as needed
        return "No '1' values in the list"


def random_index_of_zero(lst):
    """Get indices where the value is 0."""
    one_indices = [i for i, value in enumerate(lst) if value == 0]
    # Check if there is at least one '0' in the list
    if one_indices:
        # Randomly choose an index where the value is 0
        random_index = random.choice(one_indices)
        return random_index
    else:
        # If there are no '1' values in the list, return an appropriate message or handle it as needed
        return "No '0' values in the list"


def test_move():
    """Test move correctly in ingenious_base.

    Returns: True or False
    """
    ig_env = MOIngenious(num_players=2, init_draw=2, num_colors=2, board_size=8)
    ig_env.reset()
    print(ig_env.game.board_array, "nweowjrowhafhif!!!!!!!!!")
    flag = True

    # action map insist the same with index map
    for i in ig_env.game.action_index_map:
        h = ig_env.game.action_map.get(ig_env.game.action_index_map[i])
        if h is None or h != i:
            flag = False
            break
    # check legal move
    index = random_index_of_one(ig_env.game.masked_action)
    h1, h2, card = ig_env.game.action_index_map[index]
    x1, y1 = Hex2ArrayLocation(h1, ig_env.game.board_size)
    x2, y2 = Hex2ArrayLocation(h2, ig_env.game.board_size)

    if ig_env.game.board_array[x1][y1] != 0 or ig_env.game.board_array[x2][y2] != 0:
        print("reason1")
        flag = False
        return flag

    ag = ig_env.agent_selection
    c1, c2 = ig_env.game.p_tiles[ag][card]

    ig_env.game.set_action_index(index)

    ag = ig_env.agent_selection
    if ig_env.game.board_array[x1][y1] != c1 or ig_env.game.board_array[x2][y2] != c2:
        flag = False
        print("reason2")
        return flag

    # check illegal move : put somewhere not allowed
    index = random_index_of_zero(ig_env.game.masked_action)
    if ig_env.game.set_action_index(index):
        print("reason3")
        flag = False
        return flag

    # check illegal move : put some tile out of hand
    index = random_index_of_one(ig_env.game.masked_action)

    ag = ig_env.game.agents[ig_env.game.agent_selector]
    h1, h2, card = ig_env.game.action_index_map[index]
    ig_env.game.p_tiles[ag].clear()

    if ig_env.game.set_action_index(index):
        print("reason4")
        flag = False
        return flag
    return flag


def test_step():
    """Test move correctly in ingenious_base.

    Returns: True or False
    """
    ig_env = MOIngenious(num_players=2, init_draw=2, num_colors=2, board_size=8)
    ig_env.reset()
    flag = True

    # check legal step
    ag = ig_env.agent_selection

    obs = ig_env.observe(ag)
    masked_act_list = obs["action_mask"]
    index = random_index_of_one(masked_act_list)
    h1, h2, card = ig_env.game.action_index_map[index]
    x1, y1 = Hex2ArrayLocation(h1, ig_env.game.board_size)
    x2, y2 = Hex2ArrayLocation(h2, ig_env.game.board_size)

    if ig_env.game.board_array[x1][y1] != 0 or ig_env.game.board_array[x2][y2] != 0:
        print("reason1")
        flag = False
        return flag
    ag = ig_env.agent_selection
    c1, c2 = ig_env.game.p_tiles[ag][card]

    ig_env.step(index)

    ag = ig_env.agent_selection
    if ig_env.game.board_array[x1][y1] != c1 or ig_env.game.board_array[x2][y2] != c2:
        flag = False
        print("reason2")
        return flag

    # check illegal move : put somewhere not allowed
    obs = ig_env.observe(ag)
    masked_act_list = obs["action_mask"]
    index = random_index_of_zero(masked_act_list)

    remain = len(ig_env.game.tiles_bag)
    ig_env.step(index)
    if remain != len(ig_env.game.tiles_bag):
        print("reason3")
        flag = False
        return flag

    # check illegal move : put some tile out of hand
    index = random_index_of_one(ig_env.game.masked_action)
    ag = ig_env.agent_selection
    ig_env.game.p_tiles[ag].clear()
    remain = len(ig_env.game.tiles_bag)
    ig_env.step(index)
    if remain != len(ig_env.game.tiles_bag):
        print("reason4")
        flag = False
        return flag

    # check selector

    return flag


def test_reset():
    """Use MOIngenious.reset, then check if every parameter inside ingenious_base is right.

    Returns: True or False

    """
    ig_env = MOIngenious(num_players=2, init_draw=2, num_colors=2, board_size=4)
    ig_env.reset(105)
    train(ig_env)
    ig_env.reset(110)
    flag = True
    if ig_env.game.board_array.sum() != 21:
        flag = False

    if ig_env.game.end_flag:
        flag = False
    if not ig_env.game.first_round:
        flag = False
    if ig_env.game.action_size - ig_env.game.masked_action.sum() != 6 * 3 * 2 * 2:
        flag = False
    if sum([sum(s) for s in [l.values() for l in ig_env.game.score.values()]]) != 0:
        flag = False
    if ig_env.game.agent_selector != 0:
        flag = False
    if len(ig_env.game.tiles_bag) < 100:
        flag = False
    return flag


def test_ingenious_rule():
    """Ingenious rule test in a small case setting; when game end successfully, no agent should successively play 3 times."""
    ig_env = MOIngenious(num_players=2, init_draw=2, num_colors=2, board_size=8)
    ag = -1
    sum = 0
    ig_env.reset()
    done = False
    if_exeed = True
    if_ingenious = False
    while not done:
        if ag != ig_env.agent_selection:
            sum = 0
        else:
            sum += 1
        ag = ig_env.agent_selection
        obs = ig_env.observe(ag)
        masked_act_list = obs["action_mask"]
        action = random_index_of_one(masked_act_list)
        ig_env.step(action)
        observation, reward, truncation, termination, _ = ig_env.last()
        done = truncation[ag] or termination[ag]
        if sum >= 2:
            if_exeed = False
            break
        if sum == 1:
            if_ingenious = True
            break
    return if_ingenious and if_exeed


if __name__ == "__main__":
    # ig_env = MOIngenious(num_players=2, init_draw=2, num_colors=2, board_size=8)
    # ag = ig_env.agent_selection
    # ig_env.reset()
    t1 = test_ingenious_rule()
    # ig_env.reset()
    t2 = test_reset()
    # ig_env.reset()
    t3 = test_move()
    t4 = test_step()

    if t1:
        print("Accepted: ingenious rule test")
    else:
        print("Rejected: ingenious rule test")
    if t2:
        print("Accepted: reset test")
    else:
        print("Rejected: reset test")
    if t3:
        print("Accepted: move in ingenious_base test")
    else:
        print("Rejected: move in ingenious_base test")
    if t4:
        print("Accepted: move in step test")
    else:
        print("Rejected: move in step test")

    """
    ig_env = MOIngenious(num_players=2, init_draw=2, num_colors=2, board_size=4)
    ig_env.reset(105)
    train()
    ig_env.reset(110)
    flag=True
    print(len(ig_env.game.tiles_bag) >100)

"""

    # print(ig_env.reset())
    print("precommit")
    # train()

    # for i in range(1,4)
