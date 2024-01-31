"""Adapted form of the Multiwalker problem.

From Gupta, J. K., Egorov, M., and Kochenderfer, M. (2017). Cooperative multi-agent control using
deep reinforcement learning. International Conference on Autonomous Agents and Multiagent Systems
"""

from typing_extensions import override

from pettingzoo.sisl.multiwalker.multiwalker import FPS
from pettingzoo.utils import wrappers

from momaland.envs.multiwalker.momultiwalker import MOMultiwalker
from momaland.envs.multiwalker_stability.momultiwalker_stability_base import (
    MOMultiWalkerStabilityEnv,
)
from momaland.utils.conversions import mo_aec_to_parallel


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
        **kwargs: keyword args to forward to create the `MOMultiwalkerStability` environment.

    Returns:
        A raw env.
    """
    env = MOMultiwalkerStability(**kwargs)
    return env


class MOMultiwalkerStability(MOMultiwalker):
    """An MO adaptation of the [multiwalker](https://pettingzoo.farama.org/environments/sisl/multiwalker/) environment. Additionally it has an extra `stability` objective dimension.

    ## Observation Space
    See [PettingZoo documentation](https://pettingzoo.farama.org/environments/sisl/multiwalker/#observation-space).

    ## Action Space
    The action space is a vector representing the force exerted at the 4 available joints (hips and knees), giving a continuous action space with a 4 element vector.
    The higher bound is `1`, the lower bound is `-1`.

    ## Reward Space
    The reward space is a 4D vector containing rewards for:
    - Maximizing distance traveled towards the end of the level during one step. `[-0.46, 0.46]`
    - Penalty for agent falling. `[-110, 0]`
    - Penalty for the package falling. `[-100, 0]`
    - Penalty for the package tipping. `[TODO, TODO]`

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
    }

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = MOMultiWalkerStabilityEnv(*args, **kwargs)  # override engine


if __name__ == "__main__":
    from momaland.envs.multiwalker_stability import momultiwalker_stability_v0

    _env = momultiwalker_stability_v0.env(render_mode="human")

    _env.reset()
    for agent in _env.agent_iter():
        obs, rew, term, trunc, info = _env.last()
        action = None if term or trunc else _env.action_space(agent).sample()
        _env.step(action)
    _env.close()
