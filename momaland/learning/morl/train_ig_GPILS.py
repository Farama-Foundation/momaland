"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS

from momaland.learning.morl.ig_env_factory import make_single_agent_ig_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-objectives", type=int, default=2, help="Number of objectives/item types for the IG problem.")
    args = parser.parse_args()
    seed = args.seed
    obj = args.objectives

    env = make_single_agent_ig_env(objectives=obj)
    eval_env = make_single_agent_ig_env(objectives=obj)

    ref_point = np.zeros(obj)
    print("Reference point: ", ref_point)

    agent = GPILS(
        env,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        net_arch=[64, 64],
        buffer_size=1000,
        initial_epsilon=0.5,
        final_epsilon=0.05,
        epsilon_decay_steps=7500,
        learning_starts=100,
        alpha_per=0.6,
        min_priority=0.01,
        per=False,
        use_gpi=True,
        gradient_updates=10,
        target_net_update_freq=200,
        tau=1,
        log=True,
        project_name="MOMAland-Evaluation",
        seed=seed,
    )

    timesteps_per_iter = 1000
    algo = "gpi-ls"

    agent.train(
        total_timesteps=100 * timesteps_per_iter,
        eval_env=eval_env,
        ref_point=ref_point,
        weight_selection_algo=algo,
        timesteps_per_iter=timesteps_per_iter,
    )
