"""Various wrappers for Parallel MO environments
"""

import numpy as np

class Wrapper:
    """Base class for wrappers.
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        """Provide proxy access to regular attributes of wrapped objects.
        """
        return getattr(self._env, name)

class LinearizeReward(Wrapper):
    """Convert MO reward vector into scalar SO reward value.

    `weights` represents the weights of each objective in the reward vector space.
    """

    def __init__(self, env, weights:np.ndarray):
        self.weights = weights
        super().__init__(env)

    def step(self, actions):
        """Returns a reward scalar from the reward vector.
        """
        obs, rew, term, trun, info = self._env.step(actions)
        rew = np.array([np.dot(rew[agent], self.weights) for agent in rew.keys()])
        return obs, rew, term, trun, info
