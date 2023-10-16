"""MO Multiwalker problem.

From Gupta, J. K., Egorov, M., and Kochenderfer, M. (2017). Cooperative multi-agent control using
deep reinforcement learning. International Conference on Autonomous Agents and Multiagent Systems
"""

from typing_extensions import override

import numpy as np
from pettingzoo.sisl.multiwalker.multiwalker import FPS
from pettingzoo.sisl.multiwalker.multiwalker import raw_env as pz_multiwalker
from pettingzoo.utils import wrappers

from momadm_benchmarks.envs.multiwalker.multiwalker_base import MOMultiWalkerEnv as _env
from momadm_benchmarks.utils.conversions import mo_aec_to_parallel
from momadm_benchmarks.utils.env import MOAECEnv


def env(**kwargs):
    """Returns the env in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A fully wrapped AEC env.
    """
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    return env


def parallel_env(**kwargs):
    """Returns the env in `parallel` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A fully wrapped parallel env.
    """
    env = raw_env(**kwargs)
    env = mo_aec_to_parallel(env)
    return env


def raw_env(**kwargs):
    """Returns the wrapped env in `AEC` format.

    Args:
        **kwargs: keyword args to forward to create the `MOMultiwalker` environment.

    Returns:
        A fully wrapped env.
    """
    env = MOMultiwalker(**kwargs)
    return env


class MOMultiwalker(MOAECEnv, pz_multiwalker):
    """Environment for MO Multiwalker problem domain.

    The init method takes in environment arguments and should define the following attributes:
    - possible_agents
    - action_spaces
    - observation_spaces
    - reward_spaces
    These attributes should not be changed after initialization.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "momultiwalker_v0",
        "is_parallelizable": True,
        "render_fps": FPS,
    }

    @override
    def __init__(self, *args, **kwargs):
        """Initializes the multiwalker domain.

        Keyword arguments:
        n_walkers: number of bipedal walkers in environment.
        position_noise: noise applied to agent positional sensor observations.
        angle_noise: noise applied to agent rotational sensor observations.
        forward_reward: reward applied for an agent standing, scaled by agent's x coordinate.
        fall_reward: reward applied when an agent falls down.
        shared_reward: whether reward is distributed among all agents or allocated locally.
        terminate_reward: reward applied for each fallen walker in environment.
        terminate_on_fall: toggles whether agent is done if it falls down.
        terrain_length: length of terrain in number of steps.
        max_cycles: after max_cycles steps all agents will return done.
        """
        super().__init__(*args, **kwargs)
        self.env = _env(*args, **kwargs)  # override engine
        # spaces
        self.reward_spaces = dict(zip(self.agents, self.env.reward_space))

    def reward_space(self, agent):
        """Returns the reward space for the given agent."""
        return self.reward_spaces[agent]

    @override
    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(), and step() can be called without issues.

        Args:
        seed
        options

        Returns:
        the observations for each agent
        """
        super().reset(seed)  # super
        zero_reward = np.zeros(
            self.reward_spaces["walker_0"].shape, dtype=np.float32
        )  # np.copy() makes different copies of this.
        self._cumulative_rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))
