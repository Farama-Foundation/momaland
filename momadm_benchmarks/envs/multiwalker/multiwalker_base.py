from typing_extensions import override
from pettingzoo.sisl.multiwalker.multiwalker_base import TERRAIN_LENGTH, TERRAIN_STEP, TERRAIN_STARTPAD, TERRAIN_HEIGHT, LEG_H

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

    # @property
    # def reward_space(self):
    #     return spaces.Box(low, high, shape, dtype=np.float32) # TODO what is the shape of the reward space

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
        last_rewards = [0 for _ in range(self.n_walkers)] # TODO vectorize the scalar 0

    @override
    def setup(self):
        super.setup()
        self.reward_space = [agent.reward_space for agent in self.walkers] # TODO implement reward space in MOBipedalWalker

    @override
    def reset(self):
        super.reset()
        # self.lastrewards = [0 for _ in range(self.n_walkers)] # TODO vectorize the scalar 0 value