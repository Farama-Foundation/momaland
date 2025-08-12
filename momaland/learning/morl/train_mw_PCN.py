"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN

from momaland.learning.morl.sa_env_factory import make_single_agent_mw_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-project", type=str, default="PCN-MW", help="Project name.")
    parser.add_argument("-reward", type=str, default="average", help="Reward type, sum or average.")
    args = parser.parse_args()
    seed = args.seed
    reward_type = args.reward

    env = make_single_agent_mw_env(reward_type=reward_type)
    eval_env = make_single_agent_mw_env(reward_type=reward_type)

    # env = make_single_agent_mw_env_small(reward_type=reward_type)
    # eval_env = make_single_agent_mw_env_small(reward_type=reward_type)
    project_name = args.project

    obj = 2  # Multiwalker has 2 objectives: stability and speed

    ref_point = np.full(obj, -300.0, dtype=np.float32)  # Reference point for Multiwalker
    max_return = np.full(obj, 300.0, dtype=np.float32)
    print("Reference point: ", ref_point)
    print("Reward type: ", reward_type)

    agent = PCN(
        env,
        seed=seed,
        gamma=0.99,
        scaling_factor=np.ones(obj + 1),
        learning_rate=1e-3,
        hidden_dim=256,
        batch_size=256,
        noise=0.05,
        project_name=project_name,
        experiment_name="PCN",
        log=True,
    )
    # timesteps_per_iter = 10 #int(1e4)
    agent.train(
        eval_env=eval_env,
        total_timesteps=int(15e6),
        ref_point=ref_point,
        max_return=max_return,
        max_buffer_size=500,
        num_er_episodes=10,
        num_step_episodes=10,
        num_model_updates=100,
    )
