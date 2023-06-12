from random import choices

import numpy as np

from envs.congestion import moBPD


def train():
    num_agents = 5
    mobpd_env = moBPD.parallel_env(sections=3, capacity=2, num_agents=num_agents, mode="uniform")
    observations = mobpd_env.reset()
    done = False
    while not done:
        actions = choices([moBPD.LEFT, moBPD.RIGHT, moBPD.STAY], k=num_agents)
        print(actions)
        observations, rewards, done, _ = mobpd_env.step(actions)
        print(observations)
        print(rewards)


if __name__ == "__main__":
    train()
