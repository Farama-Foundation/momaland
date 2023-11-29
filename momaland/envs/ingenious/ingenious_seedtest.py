"""Temporary check file for Ingenious environment."""

import random

import numpy as np
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
    # print(ig_env.game.board_array, "nweowjrowhafhif!!!!!!!!!")
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
    # h1, h2, card = ig_env.game.action_index_map[index]
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


def check_environment_deterministic(env1, env2, num_cycles):
    """Check that two AEC environments execute the same way."""
    env1.reset(seed=42)
    env2.reset(seed=42)
    print(env1.game.log())
    print(env2.game.log())
    # seed action spaces to ensure sampled actions are the same
    seed_action_spaces(env1)
    seed_action_spaces(env2)

    # seed observation spaces to ensure first observation is the same
    seed_observation_spaces(env1)
    seed_observation_spaces(env2)

    iter = 0
    max_env_iters = num_cycles * len(env1.agents)

    for agent1, agent2 in zip(env1.agent_iter(), env2.agent_iter()):
        assert data_equivalence(agent1, agent2), f"Incorrect agent: {agent1} {agent2}"

        obs1, reward1, termination1, truncation1, info1 = env1.last()
        obs2, reward2, termination2, truncation2, info2 = env2.last()
        print(env1.agent_selection)
        print(env2.agent_selection)
        print("after")
        print(obs1, obs2)
        assert data_equivalence(obs1, obs2), "Incorrect observation"
        assert data_equivalence(reward1, reward2), "Incorrect reward."
        assert data_equivalence(termination1, termination2), "Incorrect termination."
        assert data_equivalence(truncation1, truncation2), "Incorrect truncation."
        assert data_equivalence(info1, info2), "Incorrect info."

        if termination1 or truncation1:
            break
        print("here 1")
        mask1 = obs1.get("action_mask") if isinstance(obs1, dict) else None
        mask2 = obs2.get("action_mask") if isinstance(obs2, dict) else None
        assert data_equivalence(mask1, mask2), f"Incorrect action mask: {mask1} {mask2}"
        print("here 2")
        action1 = env1.action_space(agent1).sample(mask1)
        action2 = env2.action_space(agent2).sample(mask2)

        assert data_equivalence(action1, action2), f"Incorrect actions: {action1} {action2}"

        print("before")
        print(obs1)
        print(obs2)

        env1.step(action1)
        env2.step(action2)
        print("here 3")
        iter += 1

        if iter >= max_env_iters:
            break

    env1.close()
    env2.close()


def seed_action_spaces(env):
    """Seed action space."""
    if hasattr(env, "agents"):
        for i, agent in enumerate(env.agents):
            env.action_space(agent).seed(42 + i)


def seed_observation_spaces(env):
    """Seed obs space."""
    if hasattr(env, "agents"):
        for i, agent in enumerate(env.agents):
            env.observation_space(agent).seed(42 + i)


def data_equivalence(data_1, data_2) -> bool:
    """Assert equality between data 1 and 2, i.e observations, actions, info.

    Args:
        data_1: data structure 1
        data_2: data structure 2

    Returns:
        If observation 1 and 2 are equivalent
    """
    if type(data_1) is type(data_2):
        if isinstance(data_1, dict):
            return data_1.keys() == data_2.keys() and all(data_equivalence(data_1[k], data_2[k]) for k in data_1.keys())
        elif isinstance(data_1, (tuple, list)):
            return len(data_1) == len(data_2) and all(data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2))
        elif isinstance(data_1, np.ndarray):
            # return data_1.shape == data_2.shape and np.allclose(
            #    data_1, data_2, atol=0.00001
            # )
            return data_1.shape == data_2.shape and all(data_equivalence(data_1[k], data_2[k]) for k in range(0, len(data_1)))
        else:
            return data_1 == data_2
    else:
        return False


if __name__ == "__main__":
    ig_env = MOIngenious(num_players=2, init_draw=2, num_colors=2, board_size=8)

    """
    ig_env2 = MOIngenious(num_players=2, init_draw=2, num_colors=2, board_size=8)

    ig_env.reset()
    ig_env2.reset()

    env1 = ig_env
    env2 = ig_env2

    check_environment_deterministic(env1, env2, 100)

    """
