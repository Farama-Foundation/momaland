"""Copied and pasted from PZ, adapted to handle vectorial rewards."""

import copy
from collections import defaultdict
from typing_extensions import override

import numpy as np
from pettingzoo.utils import AgentSelector
from pettingzoo.utils.conversions import (
    ActionType,
    aec_to_parallel_wrapper,
    parallel_to_aec_wrapper,
)
from pettingzoo.utils.env import AECEnv, ParallelEnv

from momaland.utils.env import MOAECEnv, MOParallelEnv


def mo_aec_to_parallel(aec_env: AECEnv) -> ParallelEnv:
    """Converts a MO aec environment to a parallel environment.

    In the case of an existing parallel environment wrapped using a `parallel_to_aec_wrapper`, this function will return the original parallel environment.
    Otherwise, it will apply the `aec_to_parallel_wrapper` to convert the environment.
    """
    if isinstance(aec_env, parallel_to_aec_wrapper):
        return aec_env.env
    else:
        par_env = mo_aec_to_parallel_wrapper(aec_env)
        return par_env


def mo_parallel_to_aec(par_env: ParallelEnv) -> AECEnv:
    """Converts a MO Parallel environment to an AEC environment.

    In the case of an existing aec environment wrapped using a `aec_to_prallel_wrapper`, this function will return the original AEC environment.
    Otherwise, it will apply the `parallel_to_aec_wrapper` to convert the environment.
    """
    if isinstance(par_env, aec_to_parallel_wrapper):
        return par_env.aec_env
    else:
        aec_env = mo_parallel_to_aec_wrapper(par_env)
        return aec_env


class mo_aec_to_parallel_wrapper(aec_to_parallel_wrapper, MOParallelEnv):
    """Converts an AEC environment into a Parallel environment.

    Overrides PZ behavior to handle vectorial rewards. Keeping inheritance avoids code duplication and checks for instance type.
    """

    def __init__(self, aec_env):
        """Converts an MO AEC environment into a MO Parallel environment."""
        super().__init__(aec_env)

    @property
    def reward_spaces(self):
        """Returns the reward spaces of the environment."""
        return self.aec_env.reward_spaces

    @override
    def reward_space(self, agent):
        return self.aec_env.reward_spaces[agent]

    def get_central_observation_space(self):
        """Delegate returning the central observation for the environment."""
        if self.aec_env.metadata.get("central_observation"):
            return self.aec_env.get_central_observation_space()
        else:
            raise NotImplementedError(
                "This environment does not support centralised observations. Please use the AEC environment directly."
            )

    @override
    def step(self, actions):
        rewards = defaultdict(lambda: np.zeros(self.reward_space(self.aec_env.agents[0]).shape))
        terminations = {}
        truncations = {}
        infos = {}
        observations = {}
        for agent in self.aec_env.agents:
            if agent != self.aec_env.agent_selection:
                if self.aec_env.terminations[agent] or self.aec_env.truncations[agent]:
                    raise AssertionError(
                        f"expected agent {agent} got termination or truncation agent {self.aec_env.agent_selection}. Parallel environment wrapper expects all agent death (setting an agent's self.terminations or self.truncations entry to True) to happen only at the end of a cycle."
                    )
                else:
                    raise AssertionError(
                        f"expected agent {agent} got agent {self.aec_env.agent_selection}, Parallel environment wrapper expects agents to step in a cycle."
                    )
            obs, rew, termination, truncation, info = self.aec_env.last()
            self.aec_env.step(actions[agent])
            for agent in self.aec_env.agents:
                rewards[agent] += self.aec_env.rewards[agent]

        terminations = dict(**self.aec_env.terminations)
        truncations = dict(**self.aec_env.truncations)
        infos = dict(**self.aec_env.infos)
        observations = {agent: self.aec_env.observe(agent) for agent in self.aec_env.agents}
        while self.aec_env.agents and (
            self.aec_env.terminations[self.aec_env.agent_selection] or self.aec_env.truncations[self.aec_env.agent_selection]
        ):
            self.aec_env.step(None)

        self.agents = self.aec_env.agents
        return observations, rewards, terminations, truncations, infos


class mo_parallel_to_aec_wrapper(parallel_to_aec_wrapper, MOAECEnv):
    """Converts a parallel environment into an AEC environment.

    Overrides PZ behavior to handle vectorial rewards. Keeping inheritance avoids code duplication and checks for instance type.
    """

    def __init__(self, parallel_env):
        """Converts a MO parallel environment into an MO AEC environment."""
        super().__init__(parallel_env)

    @property
    def reward_spaces(self):
        """Returns the reward spaces of the environment."""
        return self.env.reward_spaces

    @override
    def reward_space(self, agent):
        return self.env.reward_spaces[agent]

    @override
    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.rewards = {agent: np.zeros(self.reward_space(agent).shape[0], dtype=np.float32) for agent in self.agents}
        self._cumulative_rewards = {
            agent: np.zeros(self.reward_space(agent).shape[0], dtype=np.float32) for agent in self.agents
        }

    @override
    def add_new_agent(self, new_agent):
        super().add_new_agent(new_agent)
        self.rewards[new_agent] = np.zeros(self.reward_space(new_agent).shape[0], dtype=np.float32)
        self._cumulative_rewards[new_agent] = np.zeros(self.reward_space(new_agent).shape[0], dtype=np.float32)

    @override
    def step(self, action: ActionType):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            del self._actions[self.agent_selection]
            assert action is None
            self._was_dead_step(action)
            return
        self._actions[self.agent_selection] = action
        if self._agent_selector.is_last():
            obss, rews, terminations, truncations, infos = self.env.step(self._actions)

            self._observations = copy.copy(obss)
            self.terminations = copy.copy(terminations)
            self.truncations = copy.copy(truncations)
            self.infos = copy.copy(infos)
            self.rewards = copy.copy(rews)
            self._cumulative_rewards = copy.copy(rews)

            env_agent_set = set(self.env.agents)

            self.agents = self.env.agents + [
                agent for agent in sorted(self._observations.keys()) if agent not in env_agent_set
            ]

            if len(self.env.agents):
                self._agent_selector = AgentSelector(self.env.agents)
                self.agent_selection = self._agent_selector.reset()

            self._deads_step_first()
        else:
            if self._agent_selector.is_first():
                self._clear_rewards()

            self.agent_selection = self._agent_selector.next()

    @override
    def _clear_rewards(self) -> None:
        MOAECEnv._clear_rewards(self)
