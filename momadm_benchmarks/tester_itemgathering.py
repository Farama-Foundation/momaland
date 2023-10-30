"""Tester for the Item Gathering environment."""

import numpy as np

from momadm_benchmarks.envs.item_gathering import item_gathering
from momadm_benchmarks.envs.item_gathering.map_utils import generate_map


def train():
    """Train a random agent on the item gathering domain."""
    test_map = np.array(
        [
            [1, 0, 4, 1, 1],
            [1, 3, 4, 5, 3],
        ]
    )

    test_map = generate_map(rows=8, columns=8, item_distribution=(4, 4, 4, 2, 2), num_agents=10)
    print(test_map)

    ig_env = item_gathering.parallel_env(
        num_timesteps=50,
        initial_map=test_map,
        randomise=True,
        render_mode=None,
    )

    print(ig_env.reset())

    done = False
    ag0 = ig_env.agents[0]
    while not done:
        actions = {}
        for agent in ig_env.agents:
            actions[agent] = ig_env.action_space(agent).sample()
        print("Actions: ", actions)
        observations, rewards, truncation, termination, _ = ig_env.step(actions)
        print("Observations: ", observations)
        print("Rewards: ", rewards)
        print("Truncation: ", truncation)
        print("Termination: ", termination)
        done = truncation[ag0] or termination[ag0]


if __name__ == "__main__":
    train()
