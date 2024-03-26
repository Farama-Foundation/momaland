"""Item Gathering environment factory."""

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
        env_map = get_map_2_O()
    elif objectives == 4:
        env_map = get_map_4_O()
    else:
        env_map = DEFAULT_MAP
    ig_env = item_gathering.parallel_env(initial_map=env_map, num_timesteps=50, randomise=False, render_mode=None)
    return CentraliseAgent(ig_env, action_mapping=True)
