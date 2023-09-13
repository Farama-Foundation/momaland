"""Map utils for the Item Gathering Environment.

TODO later - write helper function to generate maps according to user configuration
"""

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

rows = 6
columns = 6
item_distribution = (3, 3, 2)  # red, green, yellow
item_locations = "fixed"
agent_locations = "fixed"
