"""Implementation of a wrapper for the tabular version of the multi-objective beach domain problem similar to the original paper."""

import numpy as np
from gymnasium.spaces import Discrete

from momaland.envs.beach.beach import (
    MOBeachDomain,
    _global_capacity_reward,
    _global_mixture_reward,
)


class TabularMOBeachDomainWrapper(MOBeachDomain):
    """Wrapper for the MO-beach domain environment to return only the current beach section as state.

    MO-Beach domain returns 5 observations in each timestep:
        - agent type
        - section id
        - section capacity
        - section consumption
        - % of agents of current type
    In the original paper however tabular Q-learning is used and therefore only the current beach section is used as
    the state. This wrapper allows to reproduce the results of the original paper. Additionally, in the original paper
    the global rewards are always reported regardless of the used reward scheme. We compute the global rewards and
    return them in the info dict of the agents.

    From Mannion, P., Devlin, S., Duggan, J., and Howley, E. (2018). Reward shaping for knowledge-based
    multi-objective multi-agent reinforcement learning.
    """

    def __init__(self, **kwargs):
        """Initialize the wrapper.

        Initialize the wrapper and set the observation space to be a discrete space with the number of sections as
        possible states.

        Additionally, set the normalization constants for the rewards depending on the chosen reward scheme.

        Args:
            **kwargs: keyword arguments for the MO-beach domain environment
        """
        self.l_cap_min, self.l_cap_max, self.l_mix_min, self.l_mix_max = kwargs.pop("local_constants")
        self.g_cap_min, self.g_cap_max, self.g_mix_min, self.g_mix_max = kwargs.pop("global_constants")
        super().__init__(**kwargs)

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Discrete(
                        self.sections,
                    )
                ],
            )
        )

    def normalize_objective_rewards(self, reward, reward_scheme):
        """Normalize the rewards based on the provided reward scheme.

        Args:
            reward: the reward to normalize
            reward_scheme: the reward scheme to use

        Returns:
            np.array: the normalized reward
        """
        # Set the normalization constants
        if reward_scheme == "individual":
            cap_min, cap_max, mix_min, mix_max = self.l_cap_min, self.l_cap_max, self.l_mix_min, self.l_mix_max
        elif reward_scheme == "team":
            cap_min, cap_max, mix_min, mix_max = self.g_cap_min, self.g_cap_max, self.g_mix_min, self.g_mix_max
        else:
            raise ValueError(f"Unknown reward scheme: {reward_scheme}")

        # Normalize the rewards
        cap_norm = (reward[0] - cap_min) / (cap_max - cap_min)
        mix_norm = (reward[1] - mix_min) / (mix_max - mix_min)

        return np.array([cap_norm, mix_norm])

    def step(self, actions):
        """Step function of the environment.

        Intercepts the observations and returns only the section id as observation.
        Also computes the global rewards and normalizes them.

        Args:
            actions: dict of actions for each agent

        Returns:
            observations: dict of observations for each agent
            rewards: dict of rewards for each agent
            terminations: dict of terminations for each agent
            truncations: dict of truncations for each agent
            infos: dict of infos for each agent
        """
        observations, rewards, terminations, truncations, infos = super().step(actions)
        # Observations: agent type, section id, section capacity, section consumption, % of agents of current type
        # Instead we want to only extract the section id
        observations = {agent: int(obs[1]) for agent, obs in observations.items()}

        # Compute global rewards in order to report them in the info dict
        section_consumptions = np.zeros(self.sections)
        section_agent_types = np.zeros((self.sections, len(self.type_distribution)))

        for i in range(len(self.possible_agents)):
            section_consumptions[self._state[i]] += 1
            section_agent_types[self._state[i]][self._types[i]] += 1
        g_capacity = _global_capacity_reward(self.resource_capacities, section_consumptions)
        g_mixture = _global_mixture_reward(section_agent_types)
        g_capacity_norm, g_mixture_norm = self.normalize_objective_rewards(np.array([g_capacity, g_mixture]), "team")
        infos = {
            agent: {"g_cap": g_capacity, "g_mix": g_mixture, "g_cap_norm": g_capacity_norm, "g_mix_norm": g_mixture_norm}
            for agent in self.possible_agents
        }

        # Normalize the rewards
        for agent in self.possible_agents:
            rewards[agent] = self.normalize_objective_rewards(rewards[agent], self.reward_mode)

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        """Reset function of the environment.

        Intercepts the observations and returns only the section id as observation.

        Args:
            seed: seed for the environment
            options: options for the environment

        Returns:
            observations: dict of observations for each agent
            infos: dict of infos for each agent
        """
        observations, infos = super().reset(seed=seed, options=options)
        # Observations: agent type, section id, section capacity, section consumption, % of agents of current type
        # Instead we want to only extract the section id
        observations = {agent: int(obs[1]) for agent, obs in observations.items()}
        return observations, infos
