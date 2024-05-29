"""Example script for environment interaction and centralised agent wrapper."""

from momaland.envs.item_gathering.map_utils import generate_map
from momaland.utils.all_modules import mobeach_v0, moitem_gathering_v0
from momaland.utils.parallel_wrappers import CentraliseAgent


def train_sa_random(moma_env):
    """Demonstrate interaction with a random centralised agent and the environment."""
    sa_env = CentraliseAgent(moma_env)

    done = False
    while not done:
        actions = sa_env.action_space.sample()
        print("Actions: ", actions)
        observations, rewards, truncation, termination, _ = sa_env.step(actions)
        print("Observations: ", observations)
        print("Rewards: ", rewards)
        print("Truncation: ", truncation)
        done = truncation or termination


def train_random(moma_env):
    """Demonstrate interaction with a random agent and the environment."""
    done = False
    ag0 = moma_env.agents[0]
    while not done:
        actions = {}
        for agent in moma_env.agents:
            actions[agent] = moma_env.action_space(agent).sample()
        print("Actions: ", actions)
        observations, rewards, truncation, termination, _ = moma_env.step(actions)
        print("Observations: ", observations)
        print("Rewards: ", rewards)
        print("Truncation: ", truncation)
        print("Termination: ", termination)
        done = truncation[ag0] or termination[ag0]


if __name__ == "__main__":
    test_map = generate_map(rows=8, columns=8, item_distribution=(4, 4, 4, 2, 2), num_agents=10, seed=0)

    ig_env = moitem_gathering_v0.parallel_env(
        num_timesteps=50,
        initial_map=test_map,
        randomise=True,
        reward_mode="test",
        render_mode=None,
    )

    mobpd_env = mobeach_v0.parallel_env(
        sections=2,
        capacity=3,
        num_agents=10,
        type_distribution=[0.5, 0.5],
        position_distribution=[0.5, 1],
        num_timesteps=10,
        reward_scheme="local",
    )

    train_random(ig_env)
    # train_random(mobpd_env)

    # train_sa_random(ig_env)
    # train_sa_random(mobpd_env)
