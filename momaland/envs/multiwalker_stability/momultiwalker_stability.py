"""Adapted form of the Multiwalker problem.

From Gupta, J. K., Egorov, M., and Kochenderfer, M. (2017). Cooperative multi-agent control using
deep reinforcement learning. International Conference on Autonomous Agents and Multiagent Systems
"""

from typing_extensions import override

import numpy as np
from gymnasium.spaces import Box
from pettingzoo.sisl.multiwalker.multiwalker import FPS
from pettingzoo.sisl.multiwalker.multiwalker import raw_env as pz_multiwalker
from pettingzoo.utils import wrappers

from momaland.envs.multiwalker_stability.momultiwalker_stability_base import (
    MOMultiWalkerStabilityEnv as _env,
)
from momaland.utils.conversions import mo_aec_to_parallel
from momaland.utils.env import MOAECEnv


def env(**kwargs):
    """Returns the wrapped environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A fully wrapped AEC env.
    """
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    return env


def parallel_env(**kwargs):
    """Returns the wrapped env in `parallel` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A fully wrapped parallel env.
    """
    env = raw_env(**kwargs)
    env = mo_aec_to_parallel(env)
    return env


def raw_env(**kwargs):
    """Returns the environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to create the `MOMultiwalker` environment.

    Returns:
        A raw env.
    """
    env = MOMultiwalkerStability(**kwargs)
    return env


class MOMultiwalkerStability(MOAECEnv, pz_multiwalker):
    """A sister environment to [MO-Multiwalker](https://momaland.farama.org/environments/momultiwalker), which is the MO adaptation of the [Multiwalker](https://pettingzoo.farama.org/environments/sisl/multiwalker/) environment from PettingZoo.

    ## Observation Space
    See [PettingZoo documentation](https://pettingzoo.farama.org/environments/sisl/multiwalker/#observation-space).

    ## Action Space
    The action space is a vector representing the force exerted at the 4 available joints (hips and knees), giving a continuous action space with a 4 element vector.
    The higher bound is `1`, the lower bound is `-1`.

    ## Reward Space
    The reward space is a 2D vector where; the first value contains the following reward:
    - Maximizing distance traveled towards the end of the level during one step. `[-0.46, 0.46]`

    and the second value contains:
    - A penalty based on the change of angle of the package, to avoid shaking the package. `[-0.01567, 0]`

    Both these objectives are penalized with:
    - Penalty for agent falling. `[-110, 0]`
    - Penalty for the package falling. `[-100, 0]`

    ## Episode Termination
    The episode is terminated if the package is dropped. If `terminate_on_fall` is `True` (default), then environment is terminated if a single agent falls even if the package is still alive.

    ## Arguments
    See [PettingZoo documentation](https://pettingzoo.farama.org/environments/sisl/multiwalker/#arguments).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "momultiwalker_stability_v0",
        "is_parallelizable": True,
        "render_fps": FPS,
        "central_observation": True,
    }

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = _env(*args, **kwargs)  # override engine
        # spaces
        self.reward_spaces = dict(zip(self.agents, self.env.reward_space))

        self.central_observation_space = Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            # 24 is the observation space of each walker, 3 is the package observation space
            shape=((24 + 3) + (24 * (len(self.agents) - 1)),),
            dtype=np.float32,
        )

    def get_central_observation_space(self):
        """Returns the central observation space for the environment."""
        return self.central_observation_space

    def reward_space(self, agent):
        """Returns the reward space for the given agent."""
        return self.reward_spaces[agent]

    @override
    def reset(self, seed=None, options=None):
        super().reset(seed, options)  # super
        zero_reward = np.zeros(
            self.reward_spaces[self.agents[0]].shape, dtype=np.float32
        )  # np.copy() makes different copies of this.
        self._cumulative_rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))


if __name__ == "__main__":
    from momaland.envs.multiwalker_stability import momultiwalker_stability_v0

    test_env = momultiwalker_stability_v0.env(render_mode="human")

    test_env.reset()
    for agent in test_env.agent_iter():
        obs, rew, term, trunc, info = test_env.last()
        action = None if term or trunc else test_env.action_space(agent).sample()
        test_env.step(action)
    test_env.close()
