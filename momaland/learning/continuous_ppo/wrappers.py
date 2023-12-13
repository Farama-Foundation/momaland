"""Wrappers for training.

Parallel only.

TODO AEC.
"""

import os
from typing import Optional

import pandas as pd
import pettingzoo
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class RecordEpisodeStatistics(BaseParallelWrapper):
    """This wrapper will record episode statistics and print them at the end of each episode."""

    def __init__(self, env: pettingzoo.ParallelEnv):
        """This wrapper will record episode statistics and print them at the end of each episode.

        Args:
            env (env): The environment to apply the wrapper
        """
        BaseParallelWrapper.__init__(self, env)
        self.episode_rewards = {agent: 0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}

    def step(self, actions):
        """Steps through the environment, recording episode statistics."""
        obs, rews, terminateds, truncateds, infos = super().step(actions)
        for agent in self.env.possible_agents:
            self.episode_rewards[agent] += rews[agent]
            self.episode_lengths[agent] += 1
        if all(terminateds.values()) or all(truncateds.values()):
            infos["episode"] = {
                "r": self.episode_rewards,
                "l": self.episode_lengths,
            }
        return obs, rews, terminateds, truncateds, infos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment, recording episode statistics."""
        obs, info = super().reset(seed, options)
        for agent in self.env.possible_agents:
            self.episode_rewards[agent] = 0
            self.episode_lengths[agent] = 0
        return obs, info


def save_results(returns, exp_name, seed):
    """Saves the results of an experiment to a csv file.

    Args:
        returns: a list of triples (timesteps, time, episodic_return)
        exp_name: experiment name
        seed: seed of the experiment
    """
    if not os.path.exists("results"):
        os.makedirs("results")
    filename = f"results/results_{exp_name}_{seed}.csv"
    print(f"Saving results to {filename}")
    df = pd.DataFrame(returns)
    df.columns = ["Total timesteps", "Time", "Episodic return"]
    df.to_csv(filename, index=False)
