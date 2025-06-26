"""Adapted from the Pistonball environment in PettingZoo."""

from typing_extensions import override

import numpy as np
from gymnasium.spaces import Box
from pettingzoo.butterfly.pistonball.pistonball import FPS
from pettingzoo.butterfly.pistonball.pistonball import raw_env as PistonballEnv
from pettingzoo.utils import wrappers

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
        **kwargs: keyword args to forward to create the `MOPistonball` environment.

    Returns:
        A raw env.
    """
    env = MOPistonball(**kwargs)
    return env


class MOPistonball(MOAECEnv, PistonballEnv):
    """An `AEC` environment where pistons need to cooperate to move a ball towards the edge of the window.

    ## Observation Space
    The observation space is unchanged from the original Pistonball. Each piston agent’s observation is an RGB image of the two pistons (or the wall) next to the agent and the space above them.

    ## Action Space
    The action space is unchanged from the original Pistonball. The action space is a 3D vector when set to discrete: 0 to move down, 1 to stay still, and 2 to move up. When set to continuous mode, an action takes a value between -1 and 1 proportional to the amount that the pistons are raised or lowered by.

    ## Reward Space
    The reward disentangles the original components of the scalar reward in Pistonball. As such, the reward space is a 2D vector containing rewards for:
    - Maximising the distance reward. From the original documentation: "The distance component is the percentage of the initial total distance (i.e. at game-start) to the left-wall travelled in the past timestep"
    - Minimizing the time penalty.

    ## Starting State
    The ball is by default dropped at the right edge of the window. This can be changed by setting `random_drop` to `True`.

    ## Episode Termination
    The episode is terminated when the ball reaches the limit of the window.

    ## Episode Truncation
    The episode is truncated when `max_cycles` is reached. This is set to 125 by default.

    ## Arguments
    The arguments are unchanged from the original Pistonball environment.
    - `n_pistons (int, optional)`: The number of pistons in the environment. Defaults to 20.
    - `time_penalty (int, optional)`: The time penalty for not finishing the episode. Defaults to -0.1.
    - `continuous (int, optional)`: Whether to use continuous actions or not. Defaults to True.
    - `random_drop (int, optional)`: Whether to drop the ball at a random place. Defaults to True.
    - `ball_mass (int, optional)`: The mass of the ball. Defaults to 0.75.
    - `ball_friction (int, optional)`: The friction of the ball. Defaults to 0.3.
    - `ball_elasticity (int, optional)`: The elasticity of the ball. Defaults to 1.5.
    - `max_cycles (int, optional)`: The maximum number of cycles in the environment before termination. Defaults to 125.
    - `render_mode (int, optional)`: The render mode. Can be human, rgb_array or None. Defaults to None.

    ## Credits
    The code was adapted from the [original Pistonball](https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/pistonball/pistonball.py).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "mopistonball_v0",
        "is_parallelizable": True,
        "render_fps": FPS,
    }

    @override
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_dim = 2  # [ball, time]

        self.reward_spaces = {
            f"piston_{i}": Box(
                low=np.array([-100, self.time_penalty]),
                high=np.array([100, 0]),
                shape=(self.reward_dim,),
                dtype=np.float32,
            )
            for i in range(self.num_agents)
        }

    def reward_space(self, agent):
        """Return the reward space for an agent."""
        return self.reward_spaces[agent]

    @override
    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        action = np.asarray(action)
        agent = self.agent_selection
        if self.continuous:
            # action is a 1 item numpy array, move_piston expects a scalar
            self.move_piston(self.pistonList[self.agent_name_mapping[agent]], action[0])
        else:
            self.move_piston(self.pistonList[self.agent_name_mapping[agent]], action - 1)

        self.space.step(self.dt)
        if self._agent_selector.is_last():
            ball_curr_pos = self._get_ball_position()

            # A rough, first-order prediction (i.e. velocity-only) of the balls next position.
            # The physics environment may bounce the ball off the wall in the next time-step
            # without us first registering that win-condition.
            ball_predicted_next_pos = ball_curr_pos + self.ball.velocity[0] * self.dt
            # Include a single-pixel fudge-factor for the approximation.
            if ball_predicted_next_pos <= self.wall_width + 1:
                self.terminate = True

            self.draw()

            # The negative one is included since the x-axis increases from left-to-right. And, if the x
            # position decreases we want the reward to be positive, since the ball would have gotten closer
            # to the left-wall.
            reward_vec = np.zeros(self.reward_dim, dtype=np.float32)
            reward_vec[0] = -1 * (ball_curr_pos - self.ball_prev_pos) * (100 / self.distance_to_wall_at_game_start)
            if not self.terminate:
                reward_vec[1] = self.time_penalty

            self.rewards = {agent: reward_vec for agent in self.agents}
            self.ball_prev_pos = ball_curr_pos
            self.frames += 1
        else:
            self._clear_rewards()

        self.truncate = self.frames >= self.max_cycles
        # Clear the list of recent pistons for the next reward cycle
        if self.frames % self.recentFrameLimit == 0:
            self.recentPistons = set()
        if self._agent_selector.is_last():
            self.terminations = dict(zip(self.agents, [self.terminate for _ in self.agents]))
            self.truncations = dict(zip(self.agents, [self.truncate for _ in self.agents]))

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = np.zeros(self.reward_dim, dtype=np.float32)
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    @override
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed, options)
        zero_reward = np.zeros(self.reward_dim, dtype=np.float32)
        self._cumulative_rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))
