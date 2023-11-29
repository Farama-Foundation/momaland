"""Various wrappers for Parallel MO environments."""

import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class LinearizeReward(BaseParallelWrapper):
    """Convert MO reward vector into scalar SO reward value.

    `weights` represents the weights of each objective in the reward vector space for each agent.

    Example:
    >>> weights = {"agent_0": np.array([0.1, 0.9]), "agent_1": np.array([0.2, 0.8]}
    >>> env = LinearizeReward(env, weights)
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
    >>>     for idx in range(env.reward_space(agent).shape[0]):
    >>>         env = AECWrappers.NormalizeReward(env, agent, idx)
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
