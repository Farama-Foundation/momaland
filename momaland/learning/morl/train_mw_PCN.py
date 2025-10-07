"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse
from datetime import datetime

import numpy as np
from morl_baselines.multi_policy.pcn.pcn import PCN

from momaland.learning.morl.sa_env_factory import make_single_agent_mw_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-project", type=str, default="PCN-MW-small", help="Project name.")
    parser.add_argument("-reward", type=str, default="sum", help="Reward type, sum or average.")
    parser.add_argument("-walkers", type=int, default=3, help="Number of walkers in the environment.")

    args = parser.parse_args()
    seed = args.seed
    reward_type = args.reward
    n_walkers = args.walkers

    env = make_single_agent_mw_env(reward_type=reward_type, n_walkers=n_walkers)
    eval_env = make_single_agent_mw_env(reward_type=reward_type, n_walkers=n_walkers)

    env.reset()
    eval_env.reset()
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
    agent.train(
        eval_env=eval_env,
        total_timesteps=int(1e8),
        ref_point=ref_point,
        max_return=max_return,
        max_buffer_size=500,
        num_er_episodes=20,
        num_step_episodes=10,
        num_model_updates=100,
    )
    # Save the agent model and add a timestamp to the filename
    dt = datetime.now()  # for date and time
    ts = datetime.timestamp(dt)  # for timestamp
    print(f"Saving agent model with name: pcn_mw_agent_{ts}")
    agent.save(filename=f"pcn_mw_agent_{ts}")
