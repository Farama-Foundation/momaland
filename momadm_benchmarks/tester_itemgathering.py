"""Tester for the Item Gathering environment."""
from random import choices

import numpy as np

from momadm_benchmarks.envs.item_gathering import item_gathering


def train():
    """Train a random agent on the beach domain."""
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

    ig_env = item_gathering.parallel_env(
        num_timesteps=10,
        env_map=test_map,
        render_mode=None,
    )

    # ig_env = item_gathering.parallel_env(
    #     num_timesteps=10,
    #     render_mode=None,
    # )

    done = False
    while not done:
        rand_act = choices(item_gathering.ACTIONS, k=len(ig_env.agents))
        actions = {}
        for i, agent in enumerate(ig_env.agents):
            actions[agent] = rand_act[i]
        print("Actions: ", actions)
        observations, rewards, done, _, _ = ig_env.step(actions)
        print("Observations: ", observations)
        print("Rewards: ", rewards)


if __name__ == "__main__":
    train()
