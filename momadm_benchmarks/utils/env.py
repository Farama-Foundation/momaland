"""Overrides PZ types to enforce multi objective rewards."""
import warnings
from typing import Dict
from typing_extensions import override

import gymnasium
import numpy as np
from numpy.typing import NDArray
from pettingzoo import AECEnv
from pettingzoo.utils.env import AgentID, ParallelEnv


class MOAECEnv(AECEnv):
    """Overrides PZ types to enforce multi objective rewards."""

    # Reward space for each agent
    reward_spaces: Dict[AgentID, gymnasium.spaces.Space]
    rewards: Dict[AgentID, NDArray]  # Reward from the last step for each agent (MO)
    # Cumulative rewards for each agent
    _cumulative_rewards: Dict[AgentID, NDArray]

    def reward_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """Takes in agent and returns the reward space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the reward_spaces dict
        """
        warnings.warn(
            "Your environment should override the reward_space function. Attempting to use the reward_spaces dict attribute."
        )
        return self.reward_spaces[agent]

    @override
    def _clear_rewards(self) -> None:
        """Clears all items in .rewards."""
        for agent in self.rewards:
            self.rewards[agent] = np.zeros(self.reward_space(agent).shape[0], dtype=np.float32)  # type: ignore


class MOParallelEnv(ParallelEnv):
    """Overrides PZ types to enforce multi objective rewards."""

    # Reward space for each agent
    reward_spaces: Dict[AgentID, gymnasium.spaces.Space]

    def reward_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        """Takes in agent and returns the reward space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the reward_spaces dict
        """
        warnings.warn(
            "Your environment should override the reward_space function. Attempting to use the reward_spaces dict attribute."
        )
        return self.reward_spaces[agent]
