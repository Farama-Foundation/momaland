"""Adapted from the Pistonball environment in PettingZoo."""


from typing_extensions import override

import numpy as np
from gymnasium.spaces import Box
from pettingzoo.butterfly.pistonball.pistonball import FPS
from pettingzoo.butterfly.pistonball.pistonball import raw_env as PistonballEnv
from pettingzoo.utils import wrappers

from momadm_benchmarks.utils.conversions import mo_aec_to_parallel
from momadm_benchmarks.utils.env import MOAECEnv


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
        **kwargs: keyword args to forward to create the `MOPistonball` environment.

    Returns:
        A raw env.
    """
    env = MOPistonball(**kwargs)
    return env


class MOPistonball(MOAECEnv, PistonballEnv):
    """A multi-objective version of the pistonball environment."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "mopistonball_v0",
        "is_parallelizable": True,
        "render_fps": FPS,
    }

    @override
    def __init__(
        self,
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
        render_mode=None,
    ):
        super().__init__(
            n_pistons=n_pistons,
            time_penalty=time_penalty,
            continuous=continuous,
            random_drop=random_drop,
            random_rotate=random_rotate,
            ball_mass=ball_mass,
            ball_friction=ball_friction,
            ball_elasticity=ball_elasticity,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        self.reward_dim = 2
        self.reward_spaces = {
            f"piston_{i}": Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)
            for i in range(self.num_agents)
        }

    def reward_space(self, agent):
        """Return the reward space for an agent."""
        return self.reward_spaces[agent]

    @override
    def step(self, action):
        """Step the environment."""
        super().step(action)

    @override
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed, options)
        zero_reward = np.zeros(self.reward_dim, dtype=np.float32)
        self._cumulative_rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))
