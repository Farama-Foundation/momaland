"""Implementation of stateless independent Q-learners. Implemented for the multi-objective congestion game and beach domain."""

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
from gymnasium.spaces import Discrete

from momaland.envs.beach_domain.beach_domain import MOBeachDomain


# from momaland.envs.congestion_game import mocongestion_v0 as CongestionGame


class TabularMOBeachDomainWrapper(MOBeachDomain):
    """Wrapper for the MO-beach domain environment to return only the current beach section as state.

    MO-Beach domain returns 5 observations in each timestep:
        - agent type
        - section id
        - section capacity
        - section consumption
        - % of agents of current type
    In the original paper however tabular Q-learning is used and therefore only the current beach section is used as
    the state. This provide allows to compare results of the original paper:

    From Mannion, P., Devlin, S., Duggan, J., and Howley, E. (2018). Reward shaping for knowledge-based multi-objective multi-agent reinforcement learning.
    """

    def __init__(self, **kwargs):
        """Initialize the wrapper.

        Initialize the wrapper and set the observation space to be a discrete space with the number of sections as
        possible states.

        Args:
            **kwargs: keyword arguments for the MO-beach domain environment
        """
        super().__init__(**kwargs)
        super().__init__(**kwargs)
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Discrete(
                        self.sections,
                    )
                ],
            )
        )

    def step(self, actions):
        """Step function of the environment.

        Intercepts the observations and returns only the section id as observation.

        Args:
            actions: dict of actions for each agent

        Returns:
            observations: dict of observations for each agent
            rewards: dict of rewards for each agent
            terminations: dict of terminations for each agent
            truncations: dict of truncations for each agent
            infos: dict of infos for each agent
        """
        observations, rewards, truncations, terminations, infos = super().step(actions)
        # Observations: agent type, section id, section capacity, section consumption, % of agents of current type
        # Instead we want to only extract the section id
        observations = {agent: int(obs[1]) for agent, obs in observations.items()}
        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        """Reset function of the environment.

        Intercepts the observations and returns only the section id as observation.

        Args:
            seed: seed for the environment
            options: options for the environment

        Returns:
            observations: dict of observations for each agent
            infos: dict of infos for each agent
        """
        observations, infos = super().reset(seed=seed, options=options)
        # Observations: agent type, section id, section capacity, section consumption, % of agents of current type
        # Instead we want to only extract the section id
        observations = {agent: int(obs[1]) for agent, obs in observations.items()}
        return observations, infos


def compute_utlity(weights, rewards):
    """Compute the utility of a given action based on the weights and the rewards of the objectives."""
    return np.dot(weights, rewards)


class QAgent:
    """Q-learning agent."""

    def __init__(self, agent_id, n_states, n_actions):
        """Initialize the Q-agent."""
        self.agent_id = agent_id
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_values = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, state, epsilon):
        """Epsilon-greedy action selection.

        Choose the action with the highest Q-value with probability 1-epsilon otherwise choose a random action.
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.q_values[state])

    def update(self, state, new_state, action, alpha, gamma, reward):
        """Update the Q-values of the agent based on the chosen action and the new state."""
        self.q_values[state][action] += alpha * (
            reward + gamma * np.max(self.q_values[new_state]) - self.q_values[state][action]
        )


def parse_args():
    """Argument parsing for hyperparameter optimization."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument('--seed', type=int, default=1,
                        help="the seed of the experiment")
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="run in debug mode")

    # Algorithm specific parameters
    parser.add_argument('--num-iterations', type=int, default=1000,
                        help="the number of training iterations")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="the learning rate")
    parser.add_argument('--alpha-decay', type=float, default=0.99,
                        help="the learning rate decay")
    parser.add_argument('--alpha-min', type=float, default=0.0,
                        help="the minimum learning rate")
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help="the exploration rate")
    parser.add_argument('--epsilon-decay', type=float, default=0.99,
                        help="the exploration rate decay")
    parser.add_argument('--epsilon-min', type=float, default=0.0,
                        help="the minimum exploration rate")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="the discount rate")

    args = parser.parse_args()
    # fmt: on
    return args


def train(args, weights):
    """IQL scalarizing the vector reward using weights and weighted sum."""
    # Environment Initialization
    # env: ParallelEnv = CongestionGame.parallel_env()
    env = TabularMOBeachDomainWrapper()
    # TODO: are these wrappers needed?
    # env = clip_actions_v0(env)
    # env = agent_indicator_v0(env)
    obs, info = env.reset()

    agents = {
        QAgent(agent_id, env.observation_space(env.agents[0]).n, env.action_spaces[agent_id].n)
        for agent_id in env.possible_agents
    }

    # Algorithm specific parameters
    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    epsilon_min = args.epsilon_min
    alpha = args.alpha
    alpha_decay = args.alpha_decay
    alpha_min = args.alpha_min
    gamma = args.gamma

    episode_returns = []

    for current_iter in range(args.num_iterations):
        if current_iter % 100 == 0:
            print(f"iteration: {current_iter}/{args.num_iterations}")
        # Get the actions from the agents
        actions = {q_agent.agent_id: q_agent.act(obs[q_agent.agent_id], epsilon) for q_agent in agents}

        # Update the exploration rate
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        else:
            epsilon = epsilon_min

        new_obs, rew, terminateds, truncations, info = env.step(actions)
        # Update the Q-values of the agents
        for agent in agents:
            # state, new_state, action, alpha, gamma, reward
            agent.update(
                obs[agent.agent_id],
                new_obs[agent.agent_id],
                actions[agent.agent_id],
                alpha,
                gamma,
                compute_utlity(weights, rew[agent.agent_id]),
            )
        avg_obj1 = np.mean(np.array(list(rew.values()))[:, 0])
        avg_obj2 = np.mean(np.array(list(rew.values()))[:, 1])
        episode_returns.append((current_iter, time.time() - start_time, {"avg_obj1": avg_obj1, "avg_obj2": avg_obj2}))

        # Update the learning rate
        if alpha > alpha_min:
            alpha *= alpha_decay
        else:
            alpha = alpha_min

        # Check if all agents are terminated (in a stateless setting, this is always the case)
        terminated = np.logical_or(
            np.any(np.array(list(terminateds.values())), axis=-1), np.any(np.array(list(truncations.values())), axis=-1)
        )
        # In case of termination, reset the environment
        if terminated:
            env.reset()

    metric = {"returned_episode_returns": np.array(episode_returns)}
    return {"metrics": metric}


if __name__ == "__main__":
    args = parse_args()
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()
    # Train a population of agents that only care about the first objective
    metrics_obj1 = train(args, np.array([1, 0]))
    print(f"metrics: {metrics_obj1['metrics']['returned_episode_returns'][-1]}")
    # Train a population of agents that only care about the second objective
    metrics_obj2 = train(args, np.array([0, 1]))
    print(f"metrics: {metrics_obj2['metrics']['returned_episode_returns'][-1]}")

    print(f"total time: {time.time() - start_time}")
