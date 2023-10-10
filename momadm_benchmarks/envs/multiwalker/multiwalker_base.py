from typing_extensions import override
from pettingzoo.sisl.multiwalker.multiwalker_base import TERRAIN_LENGTH, TERRAIN_STEP, TERRAIN_STARTPAD, TERRAIN_GRASS, TERRAIN_HEIGHT, LEG_H, VIEWPORT_W, SCALE, WALKER_SEPERATION

from pettingzoo.sisl.multiwalker.multiwalker_base import MultiWalkerEnv as pz_multiwalker_base
from pettingzoo.sisl.multiwalker.multiwalker_base import BipedalWalker as pz_bipedalwalker

import numpy as np
from gymnasium import spaces

class MOBipedalWalker(pz_bipedalwalker):
    def __init(self,
               world,
               init_x=TERRAIN_STEP * TERRAIN_STARTPAD / 2,
               init_y=TERRAIN_HEIGHT + 2 * LEG_H,
               n_walkers=2,
               seed=None
    ):
        super.__init__(world, init_x, init_y, n_walkers, seed)

    @property
    def reward_space(self):
        """
        Reward space shape = 3 element 1D array, each element representing 1 objective.
        1. package moving forward
        2. no walkers falling
        3. package not falling
        """
        return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

class MOMultiWalkerEnv(pz_multiwalker_base):
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
        pz_multiwalker_base.__init__(self,
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
            render_mode=None
        )
        self.setup()
        self.last_rewards = [np.zeros(shape=(3,), dtype=np.float32) for _ in range(self.n_walkers)] 

    @override
    def setup(self):
        super.setup()
        self.reward_space = [agent.reward_space for agent in self.walkers]

    @override
    def reset(self): # TODO is this correct?
        obs = super.reset()
        self.last_rewards = [np.zeros(shape=(3,), dtype=np.float32) for _ in range(self.n_walkers)]
        return obs

    @override
    def scroll_subroutine(self):
        xpos = np.zeros(self.n_walkers)
        obs = []
        done = False
        rewards = [np.zeros(shape=(3,), dtype=np.float32) for _ in range(self.n_walkers)]

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
            neighbor_obs.append(
                self.np_random.normal(self.package.angle, self.angle_noise)
            )
            obs.append(np.array(walker_obs + neighbor_obs))

        package_shaping = self.forward_reward * 130 * self.package.position.x
        for agent in rewards: # move forward
            agent[0] += package_shaping - self.prev_package_shaping 
        self.prev_package_shaping = package_shaping

        self.scroll = (
            xpos.mean()
            - VIEWPORT_W / SCALE / 5
            - (self.n_walkers - 1) * WALKER_SEPERATION * TERRAIN_STEP
        )

        done = [False] * self.n_walkers
        for i, (fallen, walker) in enumerate(zip(self.fallen_walkers, self.walkers)):
            if fallen: # agent doesnt fall
                for agent in rewards:
                    agent[1] += self.fall_reward
                if self.remove_on_fall:
                    walker._destroy()
                if not self.terminate_on_fall:
                    for agent in rewards:
                        agent[1] += self.terminate_reward
                done[i] = True
        if ( # package doesnt fall
            (self.terminate_on_fall and np.sum(self.fallen_walkers) > 0)
            or self.game_over
            or self.package.position.x < 0
        ):
            for agent in rewards:
                agent[2] += self.terminate_reward
            done = [True] * self.n_walkers
        elif (
            self.package.position.x
            > (self.terrain_length - TERRAIN_GRASS) * TERRAIN_STEP
        ):
            done = [True] * self.n_walkers

        return rewards, done, obs