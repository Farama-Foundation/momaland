"""Surround environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point."""

from typing_extensions import override

import numpy as np
from gymnasium.utils import EzPickle
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
        **kwargs: keyword args to forward to create the `CrazyRLSurround` environment.

    Returns:
        A raw env.
    """
    return Surround(*args, **kwargs)


class Surround(CrazyRLBaseParallelEnv, EzPickle):
    """A `Parallel` environment where drones learn how to surround a static target point.

    ## Observation Space
    The observation space is a continuous box with the length `(num_drones + 1) * 3` where each 3 values represent the XYZ coordinates of the drones in this order:
    - the agent.
    - the target.
    - the other agents.

    Example:
    `[x_0, y_0, z_0, x_targ, y_targ, z_targ, x_1, y_1, z_1, ..., x_n, y_n, z_n]`

    ## Action Space
    The action space is a 3D speed vector representing the direction in which the agent should move.

    ## Reward Space
    The reward space is a 2D vector containing rewards for:
    - Minimizing distance towards the target
    - Maximizing average distance towards other agents (avoiding collision).

    ## Starting State
    Where `size = 3`, the initial starting positions of the agents are `[0, 0, 1], [1, 1, 1], [0, 1, 1], [2, 2, 1]` while the target position is `[1, 1, 2.5]`

    ## Episode Termination
    The episode is terminated if one of the following conditions are met:
    - 2 agents collide.
    - An agent and the target collide.
    - An agent collides with the ground.

    ## Episode Truncation
    The episode is truncated when an agent reaches 200 steps.

    ## Arguments
    - `render_mode (str, optional)`: The mode to display the rendering of the environment. Can be human or None.
    - `size (int, optional)`: Size of the area sides
    - `num_drones (int, optional)`: Amount of drones
    - `init_flying_pos (nparray[float], optional)`: 2d array containing the coordinates of the agents is a (3)-shaped array containing the initial XYZ position of the drones.
    - `init_target_location (nparray[float], optional)`: A (3)-shaped array for the XYZ position of the target.
    - `target_speed (float, optional)`: Distance traveled by the target at each timestep
    - `final_target_location (nparray[float], optional)`: Array of the final position of the moving target
    - `num_intermediate_points (int, optional)`: Number of intermediate points in the target trajectory

    ## Credits
    The code was adapted from [Felten's source](https://github.com/ffelten/CrazyRL). See also the YouTube video [here](https://www.youtube.com/watch?v=4FeTjZnpgJI&t=4s&ab_channel=FlorianFelten).
    """

    metadata = {"render_modes": ["human"], "name": "surround_v0", "is_parallelizable": True, "render_fps": FPS}

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        EzPickle.__init__(
            self,
            *args,
            **kwargs,
        )

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
