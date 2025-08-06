"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN

from momaland.learning.morl.sa_env_factory import make_single_agent_mw_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-project", type=str, default="PCN-MW", help="Project name.")
    args = parser.parse_args()
    seed = args.seed

    env = make_single_agent_mw_env()
    eval_env = make_single_agent_mw_env()
    project_name = args.project
    obj = 2  # Multiwalker has 2 objectives: stability and speed

    ref_point = np.zeros(obj)
    print("Reference point: ", ref_point)

    agent = PCN(
        env,
        seed=seed,
        gamma=0.99,
        scaling_factor=np.ones(obj + 1),
        learning_rate=1e-3,
        hidden_dim=256,
        batch_size=256,
        project_name=project_name,
        experiment_name="MOMAland-MW-Centralised",
        log=False,
    )
    timesteps_per_iter = 10000
    agent.train(
        eval_env=eval_env,
        total_timesteps=15 * timesteps_per_iter,
        ref_point=ref_point,
        num_er_episodes=20,
        num_model_updates=50,
    )
