"""Tester for the Item Gathering environment."""
from random import choices

import numpy as np

from momadm_benchmarks.envs.item_gathering import item_gathering


def train():
    """Train a random agent on the item gathering domain."""
    test_map = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 0],
            [0, 0, 4, 0, 4, 5, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 3, 3, 0, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    test_map = np.array(
        [
            [0, 0, 4, 1, 1],
            [1, 3, 4, 5, 3],
        ]
    )

    test_map = np.array(
        [
            [1, 1],
            [5, 3],
        ]
    )

    ig_env = item_gathering.parallel_env(
        num_timesteps=50,
        initial_map=test_map,
        render_mode=None,
    )

    # ig_env = item_gathering.parallel_env(
    #     num_timesteps=10,
    #     render_mode=None,
    # )

    print(ig_env.reset())

    done = False
    for _ in range(10):
        # while not done:
        rand_act = choices(list(item_gathering.ACTIONS.keys()), k=len(ig_env.agents))
        actions = {}
        for i, agent in enumerate(ig_env.agents):
            actions[agent] = rand_act[i]
        # print("Actions: ", actions)
        observations, rewards, truncation, done, _ = ig_env.step(actions)
        print("Observations: ", observations)
        print("Rewards: ", rewards)
        print("Done: ", done)


if __name__ == "__main__":
    train()
