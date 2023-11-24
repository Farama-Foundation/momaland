"""Various wrappers for Parallel MO environments."""

import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


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
