"""Temporary check file for Ingenious environment."""

# import random

import numpy as np
from ingenious import Ingenious


# from ingenious_base import Hex2ArrayLocation
# from pettingzoo.classic import hanabi_v5


# from pettingzoo.test import parallel_seed_test, seed_test


"""
def train(ig_env):
    Train a random agent on the item gathering domain.
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
    Get indices where the value is 1.
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
    Get indices where the value is 0.
    one_indices = [i for i, value in enumerate(lst) if value == 0]
    # Check if there is at least one '0' in the list
    if one_indices:
        # Randomly choose an index where the value is 0
        random_index = random.choice(one_indices)
        return random_index
    else:
        # If there are no '1' values in the list, return an appropriate message or handle it as needed
        return "No '0' values in the list"
"""


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
        env1.action_space.seed(0)
        env2.action_space.seed(3)
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
    ig_env = Ingenious(num_agents=4, rack_size=4, num_colors=4, board_size=8)

    ig_env2 = Ingenious(num_agents=4, rack_size=4, num_colors=4, board_size=8)

    env1 = ig_env
    env2 = ig_env2
    env1.reset(seed=40)
    env2.reset(seed=40)
    # env1.

    env1.game.log()
    env2.game.log()

    prev_observe1, reward1, terminated1, truncated1, info1 = env1.last()
    prev_observe2, reward2, terminated2, truncated2, info2 = env1.last()
    # action = random.choice(np.flatnonzero(prev_observe["action_mask"]).tolist())
    agent1 = env1.agent_selection
    agent2 = env1.agent_selection

    print(agent1, agent2)
    """
    print(type(env1.action_space(agent1)))
    print(sum(prev_observe1["action_mask"]))
    print(sum(prev_observe2["action_mask"]))
    action1 = env1.action_space(agent1).sample(prev_observe1["action_mask"])
    action2 = env2.action_space(agent2).sample(prev_observe2["action_mask"])
    print(action1, action2)
    print(prev_observe1["action_mask"][action1], prev_observe2["action_mask"][action2])
    # check_environment_deterministic(env1, env2, 100)

    env3 = hanabi_v5.env()
    env3.reset(seed=30)
    env4 = hanabi_v5.env()
    env4.reset(seed=30)
    agent = env3.agent_selection
    action1 = env3.action_space(agent).sample()
    action2 = env4.action_space(agent).sample()
    print(action1, action2)
    """
