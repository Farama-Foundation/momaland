"""Catch environment for Crazyflie 2. Each agent is supposed to learn to surround a common target point trying to escape."""

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
        **kwargs: keyword args to forward to create the `Catch` environment.

    Returns:
        A raw env.
    """
    return Catch(*args, **kwargs)


class Catch(CrazyRLBaseParallelEnv):
    """A Parallel Environment where drone learn how to surround a moving target trying to escape."""

    metadata = {"render_modes": ["human"], "name": "catch_v0", "is_parallelizable": True, "render_fps": FPS}

    @override
    def __init__(self, *args, target_speed=0.1, **kwargs):
        """Catch environment in CrazyRL.

        Args:
            render_mode (str, optional): The mode to display the rendering of the environment. Can be human or None.
            size (int, optional): Size of the area sides
            num_drones: amount of drones
            init_flying_pos: 2d array containing the coordinates of the agents
                is a (3)-shaped array containing the initial XYZ position of the drones.
            init_target_location: Array of the initial position of the moving target
            target_speed: Distance traveled by the target at each timestep
        """

        super().__init__(*args, **kwargs)
        self.target_speed = target_speed

    def _move_target(self):
        # mean of the agent's positions
        mean = np.array([0, 0, 0])
        for agent in self.agents:
            mean = mean + self.agent_location[agent]

        mean = mean / self.num_drones

        dist = np.linalg.norm(mean - self.target_location)
        self.target_location = self.target_location.copy()

        # go to the opposite direction of the mean of the agents
        if dist > 0.2:
            self.target_location += (self.target_location - mean) / dist * self.target_speed

        # if the mean of the agents is too close to the target, move the target in a random direction, slowly because
        # it hesitates
        else:
            self.target_location += np.random.random_sample(3) * self.target_speed * 0.1

        # if the target is out of the map, put it back in the map
        np.clip(
            self.target_location,
            [-self.size, -self.size, 0.2],
            [self.size, self.size, 3],
            out=self.target_location,
        )

    @override
    def _transition_state(self, actions):
        target_point_action = dict()
        state = self.agent_location

        # new targets
        self.previous_target = self.target_location.copy()
        self._move_target()

        for agent in self.agents:
            # Actions are clipped to stay in the map and scaled to do max 20cm in one step
            target_point_action[agent] = np.clip(
                state[agent] + actions[agent] * 0.2, [-self.size, -self.size, 0], [self.size, self.size, 3]
            )

        return target_point_action


if __name__ == "__main__":
    prll_env = Catch(render_mode="human")

    observations, infos = prll_env.reset()

    while prll_env.agents:
        actions = {
            agent: prll_env.action_space(agent).sample() for agent in prll_env.agents
        }  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = prll_env.step(actions)
        prll_env.render()
