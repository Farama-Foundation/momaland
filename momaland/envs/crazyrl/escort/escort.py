"""Escort environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point moving to one point to another."""

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
    return Escort(*args, **kwargs)


class Escort(CrazyRLBaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a moving target, going straight to one point to another."""

    metadata = {"render_modes": ["human"], "name": "escort_v0", "is_parallelizable": True, "render_fps": FPS}

    def __init__(self, *args, num_intermediate_points: int = 50, final_target_location=np.array([-2, -2, 3]), **kwargs):
        """Escort environment in CrazyRL.

        Args:
            render_mode (str, optional): The mode to display the rendering of the environment. Can be human or None.
            size (int, optional): Size of the area sides
            num_drones: amount of drones
            init_flying_pos: 2d array containing the coordinates of the agents
                is a (3)-shaped array containing the initial XYZ position of the drones.
            init_target_location: A (3)-shaped array for the XYZ position of the target.
            final_target_location: Array of the final position of the moving target
            num_intermediate_points: Number of intermediate points in the target trajectory
        """
        self.final_target_location = final_target_location

        super().__init__(*args, **kwargs)

        # There are two more ref points than intermediate points, one for the initial and final target locations
        self.num_ref_points = num_intermediate_points + 2
        # Ref is a 2d arrays for the target
        # it contains the reference points (xyz) for the target at each timestep
        self.ref: np.ndarray = np.array([self.init_target_location])

        for t in range(1, self.num_ref_points):
            self.ref = np.append(
                self.ref,
                [
                    self.init_target_location
                    + (self.final_target_location - self.init_target_location) * t / self.num_ref_points
                ],
                axis=0,
            )

    @override
    def _transition_state(self, actions):
        target_point_action = dict()
        state = self.agent_location
        # new targets
        self.previous_target = self.target_location.copy()
        if self.timestep < self.num_ref_points:
            self.target_location = self.ref[self.timestep]
        else:  # the target has stopped
            self.target_location = self.ref[-1]

        for agent in self.agents:
            # Actions are clipped to stay in the map and scaled to do max 20cm in one step
            target_point_action[agent] = np.clip(
                state[agent] + actions[agent] * 0.2, [-self.size, -self.size, 0], [self.size, self.size, 3]
            )

        return target_point_action


if __name__ == "__main__":
    prll_env = Escort(render_mode="human")

    observations, infos = prll_env.reset()

    while prll_env.agents:
        actions = {
            agent: prll_env.action_space(agent).sample() for agent in prll_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = prll_env.step(actions)
        prll_env.render()
