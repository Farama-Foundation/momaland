import numpy as np

from momaland.envs.multiwalker import momultiwalker_v0 as _env
import momaland.utils.parallel_wrappers as ParallelWrappers
import momaland.utils.aec_wrappers as AECWrappers



def parallel_test():
    env = _env.parallel_env(shared_reward=False)
    # env = ParallelWrappers.LinearizeReward(env, np.array([0.3, 0.3, 0.4]))
    env = ParallelWrappers.NormalizeReward(env, env.possible_agents[0], [0, 1, 2])
    env = ParallelWrappers.NormalizeReward(env, env.possible_agents[1], [0, 1, 2])
    env = ParallelWrappers.NormalizeReward(env, env.possible_agents[2], [0, 1, 2])

    observation, info = env.reset(seed=42)

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        for key, value in rewards.items():
            print(key, value)
        print("===")
    env.close()

def aec_test():
    env = _env.env(shared_reward=False)
    env = AECWrappers.LinearizeReward(env, np.array([0.3, 0.3, 0.4]))
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        print(reward)
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample() # this is where you would insert your policy

        env.step(action)
    env.close()

if __name__ == "__main__":
    # aec_test()
    parallel_test()