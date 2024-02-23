"""MO Gymnasium on centralised agents versions of MOMAland."""

import argparse

import numpy as np
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS

from momaland.envs.item_gathering import item_gathering
from momaland.envs.item_gathering.map_utils import DEFAULT_MAP, generate_map
from momaland.utils.parallel_wrappers import CentraliseAgent


def get_map_4_O():
    """Generate a map with 4 objectives."""
    return generate_map(rows=8, columns=8, item_distribution=(3, 4, 2, 1), num_agents=2, seed=1)


def get_map_2_O():
    """Generate a map with 2 objectives."""
    return generate_map(rows=8, columns=8, item_distribution=(4, 6), num_agents=2, seed=1)


def make_single_agent_ig_env(objectives=3):
    """Create a centralised agent environment for the Item Gathering domain."""
    if objectives == 2:
        map = get_map_2_O()
    elif objectives == 4:
        map = get_map_4_O()
    else:
        map = DEFAULT_MAP
    ig_env = item_gathering.parallel_env(initial_map=map, num_timesteps=50, randomise=False, render_mode=None)
    return CentraliseAgent(ig_env, action_mapping=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=42, help="Seed for the agent.")
    parser.add_argument("-objectives", type=int, default=3, help="Seed for the agent.")
    args = parser.parse_args()
    seed = args.seed
    obj = args.objectives

    env = make_single_agent_ig_env(objectives=obj)
    eval_env = make_single_agent_ig_env(objectives=obj)

    ref_point = np.zeros(obj)
    print(ref_point)

    agent = GPILS(
        env,
        max_grad_norm=None,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[256, 256],
        buffer_size=int(2e5),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=75000,
        learning_starts=100,
        alpha_per=0.6,
        min_priority=0.01,
        per=False,
        use_gpi=True,
        gradient_updates=10,
        target_net_update_freq=200,
        tau=1,
        log=True,
        project_name="MOMAland-Baselines",
        seed=seed,
    )

    timesteps_per_iter = 10000
    algo = "gpi-ls"

    agent.train(
        total_timesteps=15 * timesteps_per_iter,
        eval_env=eval_env,
        ref_point=ref_point,
        weight_selection_algo=algo,
        timesteps_per_iter=timesteps_per_iter,
    )
