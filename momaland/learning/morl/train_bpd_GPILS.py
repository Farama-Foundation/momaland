"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS

from momaland.learning.morl.sa_env_factory import make_single_agent_bpd_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-project", type=str, default="GPI-BPD", help="Project name.")
    args = parser.parse_args()
    seed = args.seed
    obj = 2

    env = make_single_agent_bpd_env(size="small")
    eval_env = make_single_agent_bpd_env(size="small")
    project_name = args.project

    ref_point = np.zeros(obj)

    agent = GPILS(
        env,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        net_arch=[256, 256],
        buffer_size=int(2e5),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=5000,
        learning_starts=50,
        alpha_per=0.6,
        min_priority=0.01,
        per=False,
        use_gpi=True,
        gradient_updates=10,
        target_net_update_freq=150,
        tau=1,
        log=False,
        project_name=project_name,
        seed=seed,
    )

    timesteps_per_iter = 1000
    algo = "gpi-ls"

    agent.train(
        total_timesteps=10 * timesteps_per_iter,
        eval_env=eval_env,
        ref_point=ref_point,
        weight_selection_algo=algo,
        timesteps_per_iter=timesteps_per_iter,
    )
