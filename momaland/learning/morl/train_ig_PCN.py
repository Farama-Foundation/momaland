"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN

from momaland.learning.morl.sa_env_factory import make_single_agent_ig_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-objectives", type=int, default=3, help="Number of objectives/item types for the IG problem.")
    parser.add_argument("-project", type=str, default="PCN-IG", help="Project name.")
    args = parser.parse_args()
    seed = args.seed
    obj = args.objectives

    env = make_single_agent_ig_env(objectives=obj)
    eval_env = make_single_agent_ig_env(objectives=obj)
    project_name = args.project

    ref_point = np.zeros(obj)
    if obj == 2:
        max_return = np.array([4.0, 6.0])
    elif obj == 4:
        max_return = np.array([3.0, 4.0, 2.0, 1.0])
    else:
        max_return = np.array([3.0, 3.0, 2.0])
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
        experiment_name="PCN",
        log=False,
    )
    timesteps_per_iter = 10000
    agent.train(
        eval_env=eval_env,
        total_timesteps=15 * timesteps_per_iter,
        ref_point=ref_point,
        num_er_episodes=20,
        num_model_updates=50,
        max_return=max_return,
    )
