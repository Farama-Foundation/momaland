"""Adapted from the Multiwalker problem.

From Gupta, J. K., Egorov, M., and Kochenderfer, M. (2017). Cooperative multi-agent control using
deep reinforcement learning. International Conference on Autonomous Agents and Multiagent Systems
"""

from typing_extensions import override

import numpy as np
from gymnasium import spaces
from pettingzoo.sisl.multiwalker.multiwalker_base import (
    FPS,
    LEG_H,
    SCALE,
    TERRAIN_GRASS,
    TERRAIN_HEIGHT,
    TERRAIN_LENGTH,
    TERRAIN_STARTPAD,
    TERRAIN_STEP,
    VIEWPORT_W,
    WALKER_SEPERATION,
)
from pettingzoo.sisl.multiwalker.multiwalker_base import (
    BipedalWalker as pz_bipedalwalker,
)
from pettingzoo.sisl.multiwalker.multiwalker_base import (
    MultiWalkerEnv as pz_multiwalker_base,
)


class MOBipedalWalker(pz_bipedalwalker):
    """Walker Object with the physics implemented."""

    @override
    def __init__(
        self,
        world,
        forward_reward,
        fall_reward,
        terminate_reward,
        init_x=TERRAIN_STEP * TERRAIN_STARTPAD / 2,
        init_y=TERRAIN_HEIGHT + 2 * LEG_H,
        n_walkers=2,
        seed=None,
        terrain_length=TERRAIN_LENGTH,
        terrain_step=TERRAIN_STEP,
    ):
        super().__init__(world, init_x, init_y, n_walkers, seed)
        self.forward_reward = forward_reward
        self.fall_reward = fall_reward
        self.terminate_reward = terminate_reward
        self.terrain_length = terrain_length
        self.terrain_step = terrain_step

    @property
    def reward_space(self):
        """Reward space shape = 3 element 1D array, each element representing 1 objective.

        1. package moving forward.
        2. no walkers falling.
        3. package not falling.
        """
        return spaces.Box(
            low=np.array(
                [-(self.terrain_step * self.forward_reward), self.fall_reward + self.terminate_reward, self.terminate_reward]
            ),
            high=np.array([self.terrain_step * self.forward_reward, 0, 0]),
            shape=(3,),
            dtype=np.float32,
        )


class MOMultiWalkerEnv(pz_multiwalker_base):
    """Multiwalker problem domain environment engine.

    Deals with the simulation of the environment.
    """

    @override
    def __init__(
        self,
        n_walkers=3,
        position_noise=1e-3,
        angle_noise=1e-3,
        forward_reward=1.0,
        terminate_reward=-100.0,
        fall_reward=-10.0,
        shared_reward=True,
        terminate_on_fall=True,
        remove_on_fall=True,
        terrain_length=TERRAIN_LENGTH,
        max_cycles=500,
        render_mode=None,
    ):
        super().__init__(
            n_walkers=n_walkers,
            position_noise=position_noise,
            angle_noise=angle_noise,
            forward_reward=forward_reward,
            terminate_reward=terminate_reward,
            fall_reward=fall_reward,
            shared_reward=shared_reward,
            terminate_on_fall=terminate_on_fall,
            remove_on_fall=remove_on_fall,
            terrain_length=terrain_length,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        self.setup()
        self.last_rewards = [np.zeros(shape=(3,), dtype=np.float32) for _ in range(self.n_walkers)]

    @override
    def setup(self):
        """Continuation of the `__init__`."""
        super().setup()
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.walkers = [
            MOBipedalWalker(
                self.world,
                self.forward_reward,
                self.fall_reward,
                self.terminate_reward,
                init_x=sx,
                init_y=init_y,
                seed=self.seed_val,
            )
            for sx in self.start_x
        ]
        self.reward_space = [agent.reward_space for agent in self.walkers]

    @override
    def reset(self):
        obs = super().reset()
        self.last_rewards = [np.zeros(shape=(3,), dtype=np.float32) for _ in range(self.n_walkers)]
        return obs

    @override
    def step(self, action, agent_id, is_last):
        # action is array of size 4
        action = action.reshape(4)
        assert self.walkers[agent_id].hull is not None, agent_id
        self.walkers[agent_id].apply_action(action)
        if is_last:
            self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
            rewards, done, mod_obs = self.scroll_subroutine()
            self.last_obs = mod_obs
            global_reward = np.mean(rewards, axis=0)  # modified shared MO rewards
            local_reward = rewards * self.local_ratio
            self.last_rewards = global_reward * (1.0 - self.local_ratio) + local_reward * self.local_ratio
            self.last_dones = done
            self.frames = self.frames + 1

        if self.render_mode == "human":
            self.render()

    @override
    def scroll_subroutine(self):
        """This is the step engine of the environment.

        Here we have vectorized the reward math from the PettingZoo env to be multi-objective.
        """
        xpos = np.zeros(self.n_walkers)
        obs = []
        done = False
        rewards = np.array([np.zeros(shape=(3,), dtype=np.float32) for _ in range(self.n_walkers)])

        for i in range(self.n_walkers):
            if self.walkers[i].hull is None:
                obs.append(np.zeros_like(self.observation_space[i].low))
                continue
            pos = self.walkers[i].hull.position
            x, y = pos.x, pos.y
            xpos[i] = x

            walker_obs = self.walkers[i].get_observation()
            neighbor_obs = []
            for j in [i - 1, i + 1]:
                # if no neighbor (for edge walkers)
                if j < 0 or j == self.n_walkers or self.walkers[j].hull is None:
                    neighbor_obs.append(0.0)
                    neighbor_obs.append(0.0)
                else:
                    xm = (self.walkers[j].hull.position.x - x) / self.package_length
                    ym = (self.walkers[j].hull.position.y - y) / self.package_length
                    neighbor_obs.append(self.np_random.normal(xm, self.position_noise))
                    neighbor_obs.append(self.np_random.normal(ym, self.position_noise))
            xd = (self.package.position.x - x) / self.package_length
            yd = (self.package.position.y - y) / self.package_length
            neighbor_obs.append(self.np_random.normal(xd, self.position_noise))
            neighbor_obs.append(self.np_random.normal(yd, self.position_noise))
            neighbor_obs.append(self.np_random.normal(self.package.angle, self.angle_noise))
            obs.append(np.array(walker_obs + neighbor_obs))

        # Below this point is the MO reward computation. Above this point is the original PZ code.
        package_shaping = self.forward_reward * self.package.position.x
        rewards[:, 0] = package_shaping - self.prev_package_shaping  # obj1: move forward
        self.prev_package_shaping = package_shaping

        self.scroll = xpos.mean() - VIEWPORT_W / SCALE / 5 - (self.n_walkers - 1) * WALKER_SEPERATION * TERRAIN_STEP

        done = [False] * self.n_walkers
        for i, (fallen, walker) in enumerate(zip(self.fallen_walkers, self.walkers)):
            if fallen:  # obj2: agent does not fall
                rewards[i, 1] = self.fall_reward  # not all, only the one that fell
                if self.remove_on_fall:
                    walker._destroy()
                if self.terminate_on_fall:
                    rewards[:, 1] += self.terminate_reward
                done[i] = True

        if self.terminate_on_fall and np.sum(self.fallen_walkers) > 0:  # terminate_on_fall global termination
            done = [True] * self.n_walkers

        if self.game_over or self.package.position.x < 0:  # obj3: package doesn't fall
            done = [True] * self.n_walkers
            rewards[:, 2] = self.terminate_reward

        elif self.package.position.x > (self.terrain_length - TERRAIN_GRASS) * TERRAIN_STEP:
            done = [True] * self.n_walkers

        return rewards, done, obs
