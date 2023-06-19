"""Main file for showing how to run agents."""
from random import choices

from envs.beach_domain import beach_domain


def train():
    """Train a random agent on the beach domain."""
    num_agents = 5
    # mobpd_env = beach_domain.parallel_env(sections=3, capacity=2, num_agents=num_agents, type_distribution=[0.5, 0.5])
    # mobpd_env = beach_domain.parallel_env(sections=3, capacity=2, num_agents=num_agents, type_distribution=[0.333, 0.333, 0.334])

    # mobpd_env = beach_domain.parallel_env(
    #     sections=3, capacity=2, num_agents=num_agents, type_distribution=[0.333, 0.333, 0.334],
    #     position_distribution=[0.5, 0.5, 1]
    # )

    mobpd_env = beach_domain.parallel_env(
        sections=3, capacity=2, num_agents=num_agents, type_distribution=[0.5, 0.5],
        position_distribution=[0.5, 0.5, 1], num_timesteps=1, reward_scheme="global"
    )

    done = False
    while not done:
        actions = choices([beach_domain.LEFT, beach_domain.RIGHT, beach_domain.STAY], k=num_agents)
        print("Actions: ", actions)
        observations, rewards, done, _ = mobpd_env.step(actions)
        print("Observations: ", observations)
        print("Rewards: ", rewards)


if __name__ == "__main__":
    train()
