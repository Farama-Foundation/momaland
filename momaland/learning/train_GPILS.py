"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS

from momaland.envs.item_gathering import item_gathering
from momaland.utils.parallel_wrappers import CentraliseAgent


def make_single_agent_ig_env():
    """Create a centralised agent environment for the Item Gathering domain."""
    ig_env = item_gathering.parallel_env(num_timesteps=50, randomise=False, render_mode=None)
    return CentraliseAgent(ig_env, action_mapping=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    args = parser.parse_args()
    seed = args.seed

    env = make_single_agent_ig_env()
    eval_env = make_single_agent_ig_env()

    agent = GPILS(
        env,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[256, 256],
        buffer_size=int(2e5),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=75000,
        learning_starts=100,
        alpha_per=0.6,
        min_priority=0.01,
        per=False,
        use_gpi=True,
        gradient_updates=10,
        target_net_update_freq=200,
        tau=1,
        log=True,
        project_name="MOMAland-Baselines",
        seed=seed,
    )

    timesteps_per_iter = 10000
    algo = "gpi-ls"

    agent.train(
        total_timesteps=15 * timesteps_per_iter,
        eval_env=eval_env,
        ref_point=np.array([0.0, 0.0, 0.0]),
        weight_selection_algo=algo,
        timesteps_per_iter=timesteps_per_iter,
    )
