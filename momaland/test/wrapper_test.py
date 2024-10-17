"""Testing file for the wrappers."""

import numpy as np

import momaland.utils.aec_wrappers as AECWrappers
import momaland.utils.parallel_wrappers as ParallelWrappers


def aec_linearization_test(env_module):
    """Test to see if rewards have been scalarized for the given agents."""
    env = env_module.env()
    weights = {
        env.possible_agents[0]: np.random.dirichlet(np.ones(env.reward_space(env.possible_agents[0]).shape[0]), size=1)[0],
        env.possible_agents[1]: np.random.dirichlet(np.ones(env.reward_space(env.possible_agents[1]).shape[0]), size=1)[0],
    }  # attention: possible agent indexing for different agent obj
    env = AECWrappers.LinearizeReward(env, weights)
    env.reset(seed=42)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if agent is env.possible_agents[0] or agent is env.possible_agents[1]:
            assert isinstance(reward, np.float64), "Returned reward should be a scalar value of np.float64 type."
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()  # this is where you would insert your policy
        env.step(action)
    env.close()


def aec_normalization_test(env_module):
    """Soft test unit. No assertions.

    This test unit is used to check if the API breaks when the wrapper is
    applied, and is also used as an example on how to use the wrapper.

    This code can be taken as example on how to build the `weights` dict.
    """
    env = env_module.env()
    for agent in env.possible_agents:
        for idx in range(env.reward_space(agent).shape[0]):
            env = AECWrappers.NormalizeReward(env, agent, idx)
    env.reset(seed=42)
    for agent in env.agent_iter():
        observation, rewards, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()  # this is where you would insert your policy
        env.step(action)
    env.close()


def parallel_linearization_test(env_module):
    """Test to see if rewards have been scalarized for the given agents."""
    env = env_module.parallel_env()
    weights = {
        env.possible_agents[0]: np.random.dirichlet(np.ones(env.reward_space(env.possible_agents[0]).shape[0]), size=1)[0],
        env.possible_agents[1]: np.random.dirichlet(np.ones(env.reward_space(env.possible_agents[1]).shape[0]), size=1)[0],
    }  # attention: possible_agent indexing for different agent obj
    env = ParallelWrappers.LinearizeReward(env, weights)
    observations, infos = env.reset(seed=42)
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        assert isinstance(rewards[env.possible_agents[0]], np.float64) and isinstance(
            rewards[env.possible_agents[1]], np.float64
        ), "Returned reward should be a scalar value of np.float64 type."
    env.close()


def parallel_normalization_test(env_module):
    """Soft test unit. No assertions.

    This test unit is used to check if the API breaks when the wrapper is
    applied, and is also used as an example on how to use the wrapper.
    Reward bounds are not checked as RMS can be out-of-bounds

    This code can be taken as example on how to build the `weights` dict.
    """
    env = env_module.parallel_env()
    for agent in env.possible_agents:
        for idx in range(env.reward_space(agent).shape[0]):
            env = ParallelWrappers.NormalizeReward(env, agent, idx)
    observations, infos = env.reset(seed=42)
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()


def parallel_centralized_agent_test(env_module):
    """Soft test unit. No assertions.

    This test unit is used to check if the API breaks when the central agent wrapper is applied.
    """
    env = env_module.parallel_env()
    env = ParallelWrappers.CentraliseAgent(env)
    observation, info = env.reset(seed=42)
    done = False
    while done:
        actions = {agent: env.action_space[agent].sample() for agent in env.agents}
        observation, reward, truncation, termination, info = env.step(actions)
        done = truncation or termination
    env.close()


def aec_test(env_module):
    """Testing for the following wrappers for AEC:
    - Normalization
    - Linear Scalarization
    """
    aec_normalization_test(env_module)
    aec_linearization_test(env_module)
    print("Passed AEC wrapper test")


def parallel_test(env_module):
    """Testing for the following wrappers for Parallel:
    - Normalization
    - Linear Scalarization
    """
    parallel_normalization_test(env_module)
    parallel_linearization_test(env_module)
    print("Passed Parallel wrapper test")


def central_agent_test(env_module):
    """Testing for the CentralisedAgent wrapper for Parallel."""
    parallel_centralized_agent_test(env_module)
    print("Passed Centralized Agent Parallel wrapper test")


def wrapper_test(env_module):
    """Wrapper testing for AEC and Parallel environments."""
    aec_test(env_module)
    parallel_test(env_module)
    central_agent_test(env_module)
