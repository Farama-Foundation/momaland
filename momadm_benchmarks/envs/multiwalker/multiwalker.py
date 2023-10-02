import numpy as np

from typing_extensions import override

from momadm_benchmarks.utils.env import MOAECEnv

from pettingzoo.sisl.multiwalker.multiwalker import raw_env as pz_multiwalker

from momadm_benchmarks.envs.multiwalker.multiwalker_base import MOMultiWalkerEnv as _env
from pettingzoo.utils import wrappers

def env(**kwargs):
    env = mo_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class mo_env(MOAECEnv, pz_multiwalker):
    @override
    def __init__(self, *args, **kwargs):
        pz_multiwalker().__init__(self, *args, **kwargs)
        self.env = _env(*args, **kwargs) #override engine
        #spaces
        self.reward_spaces = dict(zip(self.agents, self.env.reward_space))
    
    def reward_space(self, agent):
        return self.reward_spaces[agent]
        
    @override
    def reset(self, seed=None, options=None):
        pz_multiwalker.reset() # super
        zero_reward:np.ndarray
        for agent in self.agents:
            zero_reward = np.zeros(self.reward_space(agent).shape[0], dtype=np.float32)
            break
        self._cumulative_rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents])) # CHECK check copy https://numpy.org/doc/stable/reference/generated/numpy.copy.html
        self.rewards = dict(zip(self.agents, [zero_reward.copy() for _ in self.agents]))

    # TODO
    @override
    def step(self, action):
        pass