"""Surround environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point."""
from typing_extensions import override

import numpy as np
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper

from momaland.envs.crazyrl.crazyRL_base import FPS, CrazyRLBaseParallelEnv
from momaland.utils.conversions import mo_parallel_to_aec


def env(*args, **kwargs):
    """Returns the wrapped environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A wrapped AEC env.
    """
    env = raw_env(*args, **kwargs)
    env = mo_parallel_to_aec(env)
    env = AssertOutOfBoundsWrapper(env)
    return env


def parallel_env(*args, **kwargs):
    """Returns the wrapped env in `parallel` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function.

    Returns:
        A parallel env.
    """
    env = raw_env(*args, **kwargs)
    return env


def raw_env(*args, **kwargs):
    """Returns the environment in `Parallel` format.

    Args:
        **kwargs: keyword args to forward to create the `MOMultiwalker` environment.

    Returns:
        A raw env.
    """
    return Surround(*args, **kwargs)


class Surround(CrazyRLBaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a target point."""

    metadata = {"render_modes": ["human"], "name": "surround_v0", "is_parallelizable": True, "render_fps": FPS}

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def _transition_state(self, actions):
        target_point_action = dict()
        state = self.agent_location

        for agent in self.agents:
            # Actions are clipped to stay in the map and scaled to do max 20cm in one step
            target_point_action[agent] = np.clip(
                state[agent] + actions[agent] * 0.2, [-self.size, -self.size, 0], [self.size, self.size, 3]
            )

        return target_point_action


if __name__ == "__main__":
    prll_env = Surround(render_mode="human")

    observations, infos = prll_env.reset()

    while prll_env.agents:
        actions = {
            agent: prll_env.action_space(agent).sample() for agent in prll_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = prll_env.step(actions)
        prll_env.render()
