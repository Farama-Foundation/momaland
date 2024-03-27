"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN

from momaland.learning.morl.sa_env_factory import make_single_agent_bpd_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-project", type=str, default="PCN-BPD", help="Project name.")
    args = parser.parse_args()
    seed = args.seed
    obj = 2

    env = make_single_agent_bpd_env(size="small")
    eval_env = make_single_agent_bpd_env(size="small")
    project_name = args.project

    ref_point = np.zeros(obj)
    max_return = np.array([35.0, 15.0])
    print("Reference point: ", ref_point)

    agent = PCN(
        env,
        seed=seed,
        gamma=1,
        scaling_factor=np.ones(obj + 1),
        learning_rate=1e-3,
        hidden_dim=256,
        batch_size=256,
        project_name=project_name,
        experiment_name="PCN",
        log=True,
    )
    timesteps_per_iter = 1000
    agent.train(
        eval_env=eval_env,
        total_timesteps=5 * timesteps_per_iter,
        ref_point=ref_point,
        num_er_episodes=20,
        num_model_updates=50,
        max_return=max_return,
    )
