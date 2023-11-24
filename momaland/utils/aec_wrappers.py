"""Various wrappers for AEC MO environments."""

import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd
from pettingzoo.utils.wrappers.base import BaseWrapper


class LinearizeReward(BaseWrapper):
    """Convert MO reward vector into scalar SO reward value.

    `weights` represents the weights of each objective in the reward vector space.
    """

    def __init__(self, env, weights: dict):
        """Reward linearization class initializer.

        Args:
            env: base env to add the wrapper on.
            weights: a dict where keys are agents and values are vectors representing the weights of their rewards.
        """
        self.weights = weights
        super().__init__(env)

    def last(self, observe: bool = True):
        """Returns a reward scalar from the reward vector."""
        observation, rewards, termination, truncation, info = self.env.last(observe=observe)
        if self.env.agent_selection in list(self.weights.keys()):
            rewards = np.array([np.dot(rewards, self.weights[self.env.agent_selection])])
        return observation, rewards, termination, truncation, info


class NormalizeReward(BaseWrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env,
        indices,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env: The environment to apply the wrapper
            indices: a dict with the agent names in the keys and the indices for the the rewards that should be normalized in the values.
            epsilon: A stability parameter
            gamma: The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.indices = indices
        # TODO move self.returns definition up here using `reward_space` once the BaseParallelWrapper attribute shadowing issue is solved
        self.return_rms = RunningMeanStd(shape=())
        self.returns = 0
        self.gamma = gamma
        self.epsilon = epsilon

    def last(self, observe: bool = True):
        """Steps through the environment, normalizing the rewards returned."""
        observation, rewards, termination, truncation, info = self.env.last(observe)
        agent = self.env.agent_selection
        if agent in list(self.indices.keys()):
            reward = np.array(rewards).copy()
            self.returns = self.returns * self.gamma * (1 - termination) + reward
            for i in self.indices[agent]:  # rewards that should be normalized
                reward[i] = self.normalize(reward[i], i)
            rewards = reward
        return observation, rewards, termination, truncation, info

    def normalize(self, rews, i):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
