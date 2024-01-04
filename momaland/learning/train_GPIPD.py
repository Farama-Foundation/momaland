"""MO Gymnasium on centralised agents versions of MOMAland."""

import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD

from momaland.envs.item_gathering import item_gathering
from momaland.utils.parallel_wrappers import CentraliseAgent


def make_single_agent_ig_env():
    """Create a centralised agent environment for the Item Gathering domain."""
    ig_env = item_gathering.parallel_env(num_timesteps=50, randomise=True, render_mode=None)
    return CentraliseAgent(ig_env, action_mapping=True)


if __name__ == "__main__":
    env = make_single_agent_ig_env()
    eval_env = make_single_agent_ig_env()

    gpi_pd = False
    g = 1000

    agent = GPIPD(
        env,
        num_nets=2,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=128,
        net_arch=[256, 256, 256, 256],
        buffer_size=int(2e5),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        learning_starts=100,
        alpha_per=0.6,
        min_priority=0.01,
        per=gpi_pd,
        gpi_pd=gpi_pd,
        use_gpi=True,
        gradient_updates=g,
        target_net_update_freq=200,
        tau=1,
        dyna=gpi_pd,
        dynamics_uncertainty_threshold=1.5,
        dynamics_net_arch=[256, 256, 256],
        dynamics_buffer_size=int(1e5),
        dynamics_rollout_batch_size=25000,
        dynamics_train_freq=lambda t: 250,
        dynamics_rollout_freq=250,
        dynamics_rollout_starts=5000,
        dynamics_rollout_len=1,
        real_ratio=0.5,
        log=False,
        project_name="MOMAland-Baselines",
        experiment_name="GPI-PD-IG",
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
