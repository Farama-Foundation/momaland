"""Testing file for the wrappers."""
import momaland.utils.aec_wrappers as AECWrappers
import momaland.utils.parallel_wrappers as ParallelWrappers
from momaland.envs.multiwalker import momultiwalker_v0 as _env


# def parallel_test():
#     """Full ParallelEnv lifecycle for testing wrappers."""
#     env = _env.parallel_env(shared_reward=False)
#     weights = {env.possible_agents[0]: [0.2, 0.6, 0.2], env.possible_agents[2]: [0.1, 0.3, 0.6]}
#     indices = {env.possible_agents[0]: [0], env.possible_agents[1]: [0, 1], env.possible_agents[2]: [0]}
#     env = ParallelWrappers.LinearizeReward(env, weights)
#     env = ParallelWrappers.NormalizeReward(env, indices)

#     _, _ = env.reset(seed=42)
#     while env.agents:
#         # print(env.agents)
#         # this is where you would insert your policy
#         actions = {agent: env.action_space(agent).sample() for agent in env.agents}

#         _, rewards, _, _, _ = env.step(actions)
#         for key, value in rewards.items():
#             print(key, value)
#         print("===")
#     env.close()


# def aec_test():
#     """Full AECEnv lifecycle for testing wrappers."""
#     env = _env.env(shared_reward=False)
#     weights = {env.possible_agents[0]: [0.2, 0.6, 0.2], env.possible_agents[2]: [0.1, 0.3, 0.6]}
#     indices = {env.possible_agents[0]: [0], env.possible_agents[1]: [0, 1], env.possible_agents[2]: [0]}
#     env = AECWrappers.LinearizeReward(env, weights)
#     env = AECWrappers.NormalizeReward(env, indices)

#     env.reset(seed=42)

#     for agent in env.agent_iter():
#         _, reward, termination, truncation, _ = env.last()
#         print(agent, reward)
#         if termination or truncation:
#             action = None
#         else:
#             action = env.action_space(agent).sample()  # this is where you would insert your policy

#         env.step(action)
#     env.close()

def aec_normalization_test(env):
    pass

def aec_linearization_test(env):
    pass

def parallel_normalization_test(env):
    pass

def parallel_linearization_test(env):
    pass

def aec_test(env):
    """Testing for the following wrappers for AEC:
    - Normalization
    - Linear Scalarization
    """
    aec_normalization_test(env)
    aec_linearization_test(env)

def parallel_test(env):
    """Testing for the following wrappers for Parallel:
    - Normalization
    - Linear Scalarization
    """
    parallel_normalization_test(env)
    parallel_linearization_test(env)

def wrapper_test(env):
    """Wrapper testing for AEC and Parallel environments."""
    aec_test(env)
    parallel_test(env)
