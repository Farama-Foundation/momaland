"""Various wrappers for AEC MO environments
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
    """

    def __init__(self, env, weights:np.ndarray):
        self.weights = weights
        super().__init__(env)

    def last(self):
        """Returns a reward scalar from the reward vector.
        """
        obs, rew, term, trun, info = self._env.last()
        rew = np.dot(rew, self.weights)
        return obs, rew, term, trun, info
