"""Various wrappers for Parallel MO environments."""

from collections import namedtuple
from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.wrappers.utils import RunningMeanStd
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

from momaland.learning.utils import remap_actions
from momaland.utils.env import MOParallelEnv


class RecordEpisodeStatistics(BaseParallelWrapper):
    """This wrapper will record episode statistics and print them at the end of each episode."""

    def __init__(self, env):
        """This wrapper will record episode statistics and print them at the end of each episode.

        Args:
            env (env): The environment to apply the wrapper
        """
        BaseParallelWrapper.__init__(self, env)
        self.episode_rewards = {agent: 0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}

    def step(self, actions):
        """Steps through the environment, recording episode statistics."""
        obs, rews, terminateds, truncateds, infos = super().step(actions)
        for agent in self.env.possible_agents:
            self.episode_rewards[agent] += rews[agent]
            self.episode_lengths[agent] += 1
        if all(terminateds.values()) or all(truncateds.values()):
            infos["episode"] = {
                "r": self.episode_rewards,
                "l": self.episode_lengths,
            }
        return obs, rews, terminateds, truncateds, infos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment and the episode statistics."""
        obs, info = super().reset(seed, options)
        for agent in self.env.possible_agents:
            self.episode_rewards[agent] = 0
            self.episode_lengths[agent] = 0
        return obs, info


class LinearizeReward(BaseParallelWrapper):
    """Convert MO reward vector into scalar SO reward value.

    `weights` represents the weights of each objective in the reward vector space for each agent.

    Example:
        >>> weights = {"agent_0": np.array([0.1, 0.9]), "agent_1": np.array([0.2, 0.8])}
        ... env = LinearizeReward(env, weights)
    """

    def __init__(self, env, weights: dict):
        """Reward linearization class initializer.

        Args:
            env: base env to add the wrapper on.
            weights: a dict where keys are agents and values are vectors representing the weights of their rewards.
        """
        self.weights = weights
        super().__init__(env)

    def step(self, actions):
        """Returns a reward scalar from the reward vector."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for key in rewards:
            if key not in list(self.weights):
                continue
            rewards[key] = np.dot(rewards[key], self.weights[key])

        return observations, rewards, terminations, truncations, infos


class NormalizeReward(BaseParallelWrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.

    Example:
        >>> for agent in env.possible_agents:
        ...     for idx in range(env.reward_space(agent).shape[0]):
        ...         env = AECWrappers.NormalizeReward(env, agent, idx)
    """

    def __init__(
        self,
        env,
        agent,
        idx,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env: The environment to apply the wrapper
            agent: the agent whose reward will be normalized
            idx: the index of the rewards that will be normalized.
            epsilon: A stability parameter
            gamma: The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.agent = agent
        self.idx = idx
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, actions):
        """Steps through the environment, normalizing the rewards returned."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Extracts the objective value to normalize
        to_normalize = (
            rewards[self.agent][self.idx] if isinstance(rewards[self.agent], np.ndarray) else rewards[self.agent]
        )  # array vs float

        self.returns = self.returns * self.gamma * (1 - terminations[self.agent]) + to_normalize

        # Defer normalization to gym implementation
        to_normalize = self.normalize(to_normalize)

        # Injecting the normalized objective value back into the reward vector
        # array vs float
        if isinstance(rewards[self.agent], np.ndarray):
            rewards[self.agent][self.idx] = to_normalize
        else:
            rewards[self.agent] = to_normalize

        return observations, rewards, terminations, truncations, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)


class CentraliseAgent(BaseParallelWrapper):
    """This wrapper will create a central agent that observes the full state of the environment.

    The central agent will receive the concatenation of all agents' observations as its own observation (or a global
    state, if available in the environment), and a multi-objective reward vector (representing the component-wise sum of
    the individual agent rewards) as its own reward. The central agent is expected to return a vector of actions, one
    for each agent in the original environment.
    """

    def __init__(self, env: MOParallelEnv, action_mapping=False, reward_type="sum"):
        """Central agent wrapper class initializer.

        Args:
            env: The parallel environment to apply the wrapper
            action_mapping: Whether to use an action mapping to Discrete spaces of not
            reward_type: The type of reward grouping to use, either 'sum' or 'mean'
        """
        super().__init__(env)
        self.action_mapping = action_mapping
        self.unwrapped.spec = namedtuple("Spec", ["id"])
        self.unwrapped.spec.id = self.env.metadata.get("name")
        self._reward_type = reward_type
        if self.env.metadata.get("central_observation"):
            self.observation_space = env.get_central_observation_space()
            self.unwrapped.observation_space = env.get_central_observation_space()
        else:
            self.observation_space = Dict({agentID: env.observation_space(agentID) for agentID in self.possible_agents})
        # self.action_space = Dict({agentID: env.action_space(agentID) for agentID in self.possible_agents})
        # For compatibility with MORL baselines
        # Make the action space a Box space with the same bounds as the first agent's action space
        ag0_action_space = env.action_space(self.possible_agents[0])
        self.num_actions = ag0_action_space.n
        if self.action_mapping:
            self.action_space = Discrete(self.num_actions ** len(self.possible_agents))
            self.unwrapped.action_space = self.action_space
        elif self.env.metadata.get("central_observation"):
            self.action_space = Box(
                low=ag0_action_space.start,
                high=(ag0_action_space.n - 1),
                shape=(len(self.possible_agents),),
                dtype=ag0_action_space.dtype,
            )
        else:
            self.action_space = Dict({agentID: env.action_space(agentID) for agentID in self.possible_agents})
        self.reward_space = self.env.reward_space(self.possible_agents[0])
        self.unwrapped.reward_space = self.reward_space

    def step(self, actions):
        """Steps through the environment, joining the returned values for the central agent."""
        # Remake the action list into a dictionary compatible with MOMAland environments
        if self.action_mapping:
            remapped_actions = remap_actions(actions, len(self.agents), self.num_actions)
            actions = {agent: remapped_actions[i] for i, agent in enumerate(self.agents)}
        elif self.env.metadata.get("central_observation"):
            actions = {agent: actions[num] for num, agent in enumerate(self.possible_agents)}

        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        if self.env.metadata.get("central_observation"):
            observations = self.env.state().flatten()
        if self._reward_type == "sum":
            joint_reward = np.sum(list(rewards.values()), axis=0)
        else:
            joint_reward = np.mean(list(rewards.values()), axis=0)
        return (
            observations,
            joint_reward,
            np.any(list(terminations.values())),
            np.any(list(truncations.values())),
            infos,
        )

    def reset(self, seed=None):
        """Resets the environment, joining the returned values for the central agent."""
        observations, infos = self.env.reset(seed)
        if self.env.metadata.get("central_observation"):
            observations = self.env.state().flatten()
        return observations, list(infos.values())
