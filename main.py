from envs.beach_domain import beach_domain
import numpy as np
from random import choices


def train():
    num_agents = 5
    mobpd_env = beach_domain.parallel_env(sections=3, capacity=2, num_agents=num_agents, num_types=3)
    done = False
    while not done:
        actions = choices([beach_domain.LEFT, beach_domain.RIGHT, beach_domain.STAY], k=num_agents)
        print('Actions: ', actions)
        observations, rewards, done, _ = mobpd_env.step(actions)
        print('Observations: ', observations)
        print('Rewards: ', rewards)


if __name__ == "__main__":
    train()
