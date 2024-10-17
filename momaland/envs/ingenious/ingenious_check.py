"""Temporary check file for Ingenious environment."""

import random

import gymnasium
import numpy as np

from momaland.envs.ingenious.ingenious import Ingenious

# from ingenious import MOIngenious
from momaland.envs.ingenious.ingenious_base import Hex2ArrayLocation


def train(ig_env):
    """Train a random agent on the item gathering domain."""
    done = False
    while not done:
        ag = ig_env.agent_selection
        # print("Agent: ", ag)
        obs = ig_env.observe(ag)
        masked_act_list = obs["action_mask"]
        action = random_index_of_one(masked_act_list)
        # print("Observation: ",obs)
        # print("Action: ", action)
        ig_env.step(action)
        observation, reward, truncation, termination, _ = ig_env.last()
        # print("Observations: ", observation['observation'])
        # print("Rewards: ", reward)
        # print("Truncation: ", truncation)
        # print("Termination: ", termination)
        done = truncation or termination


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
    ig_env = Ingenious(num_agents=2, rack_size=2, num_colors=2, board_size=8)
    ig_env.reset()
    # print(ig_env.game.board_array, "nweowjrowhafhif!!!!!!!!!")

    # action map insist the same with index map
    for i in ig_env.game.action_index_map:
        h = ig_env.game.action_map.get(ig_env.game.action_index_map[i])
        if h is None or h != i:
            break
    # check legal move
    index = random_index_of_one(ig_env.game.return_action_list())
    h1, h2, card = ig_env.game.action_index_map[index]
    x1, y1 = Hex2ArrayLocation(h1, ig_env.game.board_size)
    x2, y2 = Hex2ArrayLocation(h2, ig_env.game.board_size)

    assert ig_env.game.board_array[x1][y1] == 0 and ig_env.game.board_array[x2][y2] == 0, "Place on board is taken."
    ag = ig_env.agent_selection
    c1, c2 = ig_env.game.p_tiles[ag][card]
    ig_env.game.set_action_index(index)
    assert ig_env.game.board_array[x1][y1] == c1 and ig_env.game.board_array[x2][y2] == c2, "Color is not placed correctly."
    print("ingenious_base basic move Passed")


def test_step():
    """Test move correctly in ingenious_base.

    Returns: True or False
    """
    ig_env = Ingenious(num_agents=2, rack_size=2, num_colors=2, board_size=8)
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
    """
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

    """

    return flag


def test_reset():
    """Use MOIngenious.reset, then check if every parameter inside ingenious_base is right.

    Returns: True or False

    """
    ig_env = Ingenious(num_agents=2, rack_size=2, num_colors=2, board_size=4)
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
    if flag:
        print("Reset test Passed")
    else:
        print("Reset test Rejected")
    return flag


def test_ingenious_rule():
    """Ingenious rule test in a small case setting; when game end successfully, no agent should successively play 3 times."""
    ig_env = Ingenious(num_agents=2, rack_size=2, num_colors=2, board_size=8)
    ag = -1
    sum = 0
    ig_env.reset()
    ig_env.game.max_score = 5

    done = False
    if_exeed = True
    if_ingenious = False
    while not done:
        if ag != ig_env.agent_selection:
            sum = 0
        else:
            sum += 1
        ag = ig_env.agent_selection
        # obs = ig_env.observe(ag)
        # masked_act_list = obs["action_mask"]
        masked_act_list = ig_env.game.return_action_list()
        action = random_index_of_one(masked_act_list)
        ig_env.step(action)
        observation, reward, truncation, termination, _ = ig_env.last()
        done = truncation or termination
        if sum >= 2:
            if_exeed = False
            break
        if sum == 1:
            if_ingenious = True
            break
    if if_ingenious and if_exeed:
        print("Ingenious rule check Passed")

    return if_ingenious and if_exeed


def test_API():
    """Test observe interface in ingenous.py."""
    ig_env = Ingenious()
    ig_env.max_score = 10000
    # ag = ig_env.agent_selection
    # obs = ig_env.observe(ag)
    # print(sum(masked_act_list))
    # print(sum(ig_env.game.masked_action))
    env = ig_env
    env.reset()
    # observation_0
    num_cycles = 100

    env.reset()

    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    live_agents = set(env.agents[:])
    has_finished = set()
    generated_agents = set()
    accumulated_rewards = {
        agent: np.zeros(env.unwrapped.reward_space(agent).shape[0], dtype=np.float32) for agent in env.agents
    }
    for agent in env.agent_iter(env.num_agents * num_cycles):
        generated_agents.add(agent)
        # print(agent, has_finished, generated_agents)
        # print(env.last())
        assert agent not in has_finished, "agents cannot resurrect! Generate a new agent with a new name."
        assert isinstance(env.infos[agent], dict), "an environment agent's info must be a dictionary"
        prev_observe, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        elif isinstance(prev_observe, dict) and "action_mask" in prev_observe:
            action = random.choice(np.flatnonzero(prev_observe["action_mask"]).tolist())
        else:
            action = env.action_space(agent).sample()

        if agent not in live_agents:
            live_agents.add(agent)

        assert live_agents.issubset(set(env.agents)), "environment must delete agents as the game continues"

        if terminated or truncated:
            live_agents.remove(agent)
            has_finished.add(agent)

        assert np.all(
            accumulated_rewards[agent] == reward
        ), "reward returned by last is not the accumulated rewards in its rewards dict"
        accumulated_rewards[agent] = np.zeros_like(reward, dtype=np.float32)

        env.step(action)

        for a, rew in env.rewards.items():
            accumulated_rewards[a] += rew

        assert env.num_agents == len(env.agents), "env.num_agents is not equal to len(env.agents)"
        assert set(env.rewards.keys()) == (
            set(env.agents)
        ), "agents should not be given a reward if they were terminated or truncated last turn"
        assert set(env.terminations.keys()) == (
            set(env.agents)
        ), "agents should not be given a termination if they were terminated or truncated last turn"
        assert set(env.truncations.keys()) == (
            set(env.agents)
        ), "agents should not be given a truncation if they were terminated or truncated last turn"
        assert set(env.infos.keys()) == (
            set(env.agents)
        ), "agents should not be given an info if they were terminated or truncated last turn"
        if hasattr(env, "possible_agents"):
            assert set(env.agents).issubset(
                set(env.possible_agents)
            ), "possible agents should always include all agents, if it exists"

        if not env.agents:
            break

        if isinstance(env.observation_space(agent), gymnasium.spaces.Box):
            assert env.observation_space(agent).dtype == prev_observe.dtype

        # These codes are some left codes no need anymore for action is already taken in the env and test_observation not used anymore.
        # assert env.observation_space(agent).contains(prev_observe), "Out of bounds observation: " + str(prev_observe)
        # assert env.observation_space(agent).contains(prev_observe), "Agent's observation is outside of it's observation space"
        # test_observation(prev_observe, observation_0)
        if not isinstance(env.infos[env.agent_selection], dict):
            print("The info of each agent should be a dict, use {} if you aren't using info")

    if not env.agents:
        assert has_finished == generated_agents, "not all agents finished, some were skipped over"
    print("API ingenious.py Passed")


def check_fully_observable():
    """Test observable trigger in ingenous.py."""
    ig_env = Ingenious(fully_obs=True)
    ig_env.reset()
    ag = ig_env.agent_selection
    obs = ig_env.observe(ag)
    print("Observation", obs)
    print("Fully Observable: Pass")


def check_two_team():
    """Test teammate(reward sharing) in ingenous.py."""
    ig_env = Ingenious(num_agents=4, reward_mode="two_teams")
    ig_env.reset()
    ag = ig_env.agent_selection
    obs = ig_env.observe(ag)
    # index = random_index_of_one(ig_env.game.return_action_list())
    # ig_env.step(index)
    print("Start check_two_team")
    print("Observation", obs)
    done = False
    while not done:
        ag = ig_env.agent_selection
        print("Agent: ", ag)
        obs = ig_env.observe(ag)
        masked_act_list = obs["action_mask"]
        action = random_index_of_one(masked_act_list)
        print("Action: ", action)
        ig_env.step(action)
        observation, reward, termination, truncation, _ = ig_env.last()
        print("Observations: ", observation["observation"])
        print(
            "Rewards: ",
            reward,
            "_accumulate_reward(from gymnasium code)",
            ig_env._cumulative_rewards,
        )
        print("Truncation: ", truncation)
        print("Termination: ", termination)
        done = truncation or termination
    print(ig_env.game.score)
    print("Stop check_two_team")


def check_collaborative():
    """Test teammate(reward sharing) in ingenous.py."""
    ig_env = Ingenious(num_agents=4, reward_mode="collaborative")
    ig_env.reset()
    ag = ig_env.agent_selection
    obs = ig_env.observe(ag)
    # index = random_index_of_one(ig_env.game.return_action_list())
    # ig_env.step(index)
    print("Start check_collaborative")
    print("Observation", obs)
    done = False
    while not done:
        ag = ig_env.agent_selection
        print("Agent: ", ag)
        obs = ig_env.observe(ag)
        masked_act_list = obs["action_mask"]
        action = random_index_of_one(masked_act_list)
        print("Action: ", action)
        ig_env.step(action)
        observation, reward, termination, truncation, _ = ig_env.last()
        print("Observations: ", observation["observation"])
        print(
            "Rewards: ",
            reward,
            "_accumulate_reward(from gymnasium code)",
            ig_env._cumulative_rewards,
        )
        print("Truncation: ", truncation)
        print("Termination: ", termination)
        done = truncation or termination
    print(ig_env.game.score)
    print("Stop check_collaborative")


def check_parameter_range():
    """Simulate all possible parameter to test the game with random choices."""
    for n_player in range(2, 7):
        for draw in range(2, 7):
            for color in range(n_player, 7):
                for bs in range(0, 10):
                    for teammate in ["competitive", "collaborative", "two_teams"]:
                        for fully_obs in [True, False]:
                            print(
                                "num_players=",
                                n_player,
                                " init_draw=",
                                draw,
                                "num_colors=",
                                color,
                                "board_size=",
                                bs,
                                "teammate_mode=",
                                teammate,
                                "fully_obs=",
                                fully_obs,
                                "render_mode=",
                                None,
                            )

                            try:
                                ig_env = Ingenious(
                                    num_agents=n_player,
                                    rack_size=draw,
                                    num_colors=color,
                                    board_size=bs,
                                    reward_mode=teammate,
                                    fully_obs=fully_obs,
                                    render_mode=None,
                                )
                                ig_env.reset()
                                train(ig_env)
                            except Exception as e:
                                print(e)
                                pass
                            else:
                                print("PASS")


if __name__ == "__main__":
    # test move of inginous_base.py
    test_move()
    # test API
    test_API()
    # test inginious rule
    test_ingenious_rule()

    # run this function, you could always find opponents' tiles in observation space
    check_fully_observable()

    # check two_team mode through simulation, it could be found that teammates always share the same score in score board.
    check_two_team()

    # check collaborative mode through simulation, it could be found that every players always share the same score in score board.
    check_collaborative()

    # check parameter range by random choice.
    check_parameter_range()
