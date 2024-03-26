"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN

from momaland.learning.morl.ig_env_factory import make_single_agent_ig_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=44, help="Seed for the agent.")
    parser.add_argument("-objectives", type=int, default=2, help="Number of objectives/item types for the IG problem.")
    parser.add_argument("-project", type=str, default="MOMAland-IG", help="Project name.")
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
    print("Reference point: ", ref_point)

    agent = PCN(
        env,
        seed=seed,
        gamma=0.99,
        scaling_factor=np.ones(obj + 1),
        learning_rate=1e-3,
        batch_size=256,
        project_name=project_name,
        experiment_name="PCN",
        log=True,
    )
    timesteps_per_iter = 10000
    agent.train(
        eval_env=eval_env,
        total_timesteps=10 * timesteps_per_iter,
        ref_point=ref_point,
        num_er_episodes=20,
        max_buffer_size=50,
        num_model_updates=50,
        max_return=max_return,
    )
