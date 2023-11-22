"""Various wrappers for AEC MO environments."""

import numpy as np
from pettingzoo.utils.wrappers.base import BaseWrapper


# class Wrapper:
#     """Base class for wrappers."""

#     def __init__(self, env):
#         """Base wrapper initialization to save the base env."""
#         self._env = env

#     def __getattr__(self, name):
#         """Provide proxy access to regular attributes of wrapped objects."""
#         return getattr(self._env, name)


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


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = m2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


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
        agent,
        indices,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env: The environment to apply the wrapper
            agent: the name of the agent in string whose reward(s) will be normalized.
            indices: a ndarray with the indices for the values in the reward vector that should be normalized.
            epsilon: A stability parameter
            gamma: The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self._env = env
        self.agent = agent
        self.indices = indices
        self.num_rewards = self._env.reward_spaces[agent].shape[0]
        self.return_rms = np.array(
            [RunningMeanStd(shape=()) for _ in range(self.num_rewards)]
        )  # separate runningmeanstd for each obj
        self.returns = 0
        self.gamma = gamma
        self.epsilon = epsilon

    def last(self, observe: bool = True):
        """Returns the last obs, rew, term, trunc, info; normalizing the rewards returned."""
        observation, reward, termination, truncation, info = self.env.last(observe=observe)
        self.returns = self.returns * self.gamma * (1 - termination) + reward
        for i in self.indices:
            reward[i] = self.normalize(reward[i], i)
        return observation, reward, termination, truncation, info

    def normalize(self, rews, i):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms[i].update(self.returns)
        return rews / np.sqrt(self.return_rms[i].var + self.epsilon)
