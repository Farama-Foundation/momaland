"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import (
    GPILSContinuousAction,
)

from momaland.learning.morl.sa_env_factory import make_single_agent_mw_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-project", type=str, default="GPI-MW", help="Project name.")
    args = parser.parse_args()
    seed = args.seed

    env = make_single_agent_mw_env()
    eval_env = make_single_agent_mw_env()
    project_name = args.project
    obj = 2  # Multiwalker has 2 objectives: stability and speed

    ref_point = np.full(obj, -300.0, dtype=np.float32)  # Reference point for Multiwalker
    print("Reference point: ", ref_point)

    agent = GPILSContinuousAction(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        net_arch=[256, 256],
        buffer_size=int(2e5),
        learning_starts=100,
        use_gpi=True,
        gradient_updates=10,
        log=True,
        project_name=project_name,
        experiment_name="GPI",
        seed=seed,
    )

    timesteps_per_iter = int(1e4)
    algo = "gpi-ls"

    agent.train(
        total_timesteps=int(1e6 * timesteps_per_iter),
        eval_env=eval_env,
        ref_point=ref_point,
        weight_selection_algo=algo,
        timesteps_per_iter=timesteps_per_iter,
    )
