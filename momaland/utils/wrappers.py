"""Various wrappers for MO environments
"""

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

    TODO
    """
