"""Various wrappers for Parallel MO environments."""

import numpy as np
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class Wrapper:
    """Base class for wrappers."""

    def __init__(self, env):
        """Base wrapper initialization to save the base env."""
        self._env = env

    def __getattr__(self, name):
        """Provide proxy access to regular attributes of wrapped objects."""
        return getattr(self._env, name)


class LinearizeReward(BaseParallelWrapper):
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

    def step(self, actions):
        """Returns a reward scalar from the reward vector."""
        observations, rewards, terminations, truncations, infos = super().step(
            actions
        )  # super.step is called to have env.agents reachable, otherwise main loop never ends
        for key in rewards.keys():
            if key not in list(self.weights.keys()):
                continue
            rewards[key] = np.array([np.dot(rewards[key], self.weights[key])])

        return observations, rewards, terminations, truncations, infos


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
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(batch_mean, batch_var, batch_count)

    def update_mean_var_count_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class NormalizeReward(BaseParallelWrapper):
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
        self.return_rms = (
            np.array(  # length of amount of agents that will be normalized + separate runningmeanstd for each obj
                [RunningMeanStd(shape=()) for _ in range(len(self.indices.keys()))]
            )
        )
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, actions):
        """Steps through the environment, normalizing the rewards returned."""
        observations, rewards, terminations, truncations, infos = super().step(actions)
        self.returns = len(
            list(rewards[list(rewards.keys())[0]])
        )  # getting the max amount of obj amongst all agents that need to be normalized
        for key, value in self.indices.items():
            if key not in rewards.keys():
                continue
            reward = np.array(rewards[key])
            self.returns = self.returns * self.gamma * (1 - terminations[key]) + reward
            for i in value:  # rewards that should be normalized
                reward[i] = self.normalize(reward[i], i)
            rewards[key] = reward
        return observations, rewards, terminations, truncations, infos

    def normalize(self, rews, i):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms[i].update(self.returns)
        return rews / np.sqrt(self.return_rms[i].var + self.epsilon)
