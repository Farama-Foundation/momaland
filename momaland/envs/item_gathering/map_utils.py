"""Map utils for the Item Gathering Environment."""

import numpy as np


DEFAULT_MAP = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 4, 0, 4, 5, 0, 0],
        [0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 3, 3, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ]
)


def randomise_map(env_map, seed=None):
    """Randomize the interior of the map (only the item positions).

    Args:
        env_map (np.ndarray): The map to randomize.
        seed (int, optional): The seed to use for the randomization. Defaults to None.

    Returns:
        np.ndarray: The randomized map.
    """
    if seed is not None:
        np.random.seed(seed)
    interior_map = env_map[2:-2, 2:-2]
    # shuffle row-wise
    interior_map = np.random.permutation(interior_map)
    # shuffle column-wise
    interior_map = np.random.permutation(interior_map.transpose()).transpose()
    env_map[2:-2, 2:-2] = interior_map
    return env_map


def generate_map(rows=8, columns=8, item_distribution=(3, 3, 2), num_agents=2, seed=None):
    """Generate a map for the Item Gathering environment, according to the original paper.

    Items are only generated in the interior of the map, i.e., not in the 2 outermost layers of the map.

    Args:
        rows (int, optional): The number of rows in the map. Defaults to 8.
        columns (int, optional): The number of columns in the map. Defaults to 8.
        item_distribution (tuple, optional): The number of items of each type to place in the map. Defaults to (3, 3, 2).
        num_agents (int, optional): The number of agents to place in the map. Defaults to 2.
        seed (int, optional): The seed to use for the randomization. Defaults to None.
    """
    if seed is not None:
        np.random.seed(seed)
    map = np.zeros((rows, columns), dtype=int)
    interior_map = map[2:-2, 2:-2]
    item_index_start = 2
    for item in item_distribution:
        for i in range(item):
            empty_locations = np.argwhere(interior_map == 0)
            assert empty_locations.shape[0] > 0, "Not enough empty locations to place items"
            # randomly place items in the interior of the map, in an empty location
            item_location = empty_locations[np.random.randint(0, empty_locations.shape[0])]
            interior_map[item_location[0], item_location[1]] = item_index_start
        item_index_start += 1
    map[2:-2, 2:-2] = interior_map
    # randomly place agents in the remaining empty cells
    for i in range(num_agents):
        empty_locations = np.argwhere(map == 0)
        assert empty_locations.shape[0] > 0, "Not enough empty locations to place agents"
        # randomly select from the available positions
        agent_location = empty_locations[np.random.randint(0, empty_locations.shape[0])]
        map[agent_location[0], agent_location[1]] = 1
    return map
