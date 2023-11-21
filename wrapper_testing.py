"""Testing file for the wrappers."""
import numpy as np

import momaland.utils.aec_wrappers as AECWrappers
import momaland.utils.parallel_wrappers as ParallelWrappers
from momaland.envs.multiwalker import momultiwalker_v0 as _env


def parallel_test():
    """Full ParallelEnv lifecycle for testing wrappers."""
    env = _env.parallel_env(shared_reward=False)
    env = ParallelWrappers.LinearizeReward(env, np.array([0.8, 0.15, 0.15]))
    env = ParallelWrappers.NormalizeReward(env, env.possible_agents[0], [0])
    env = ParallelWrappers.NormalizeReward(env, env.possible_agents[1], [0])
    env = ParallelWrappers.NormalizeReward(env, env.possible_agents[2], [0])

    _, _ = env.reset(seed=42)

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        _, rewards, _, _, _ = env.step(actions)
        for key, value in rewards.items():
            print(key, value)
        print("===")
    env.close()


def aec_test():
    """Full AECEnv lifecycle for testing wrappers."""
    env = _env.env(shared_reward=False)
    env = AECWrappers.LinearizeReward(env, np.array([0.33, 0.33, 0.33]))
    env = AECWrappers.NormalizeReward(env, env.possible_agents[0], [0])
    env = AECWrappers.NormalizeReward(env, env.possible_agents[1], [0])
    env = AECWrappers.NormalizeReward(env, env.possible_agents[2], [0])

    env.reset(seed=42)

    for agent in env.agent_iter():
        _, reward, termination, truncation, _ = env.last()
        print(agent, reward)
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()  # this is where you would insert your policy

        env.step(action)
    env.close()


if __name__ == "__main__":
    aec_test()
    # parallel_test()
