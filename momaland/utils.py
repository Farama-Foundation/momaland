"""Helper functions for the MOMAland."""
import errno
import os

import numpy as np


def u_imbalance(x):
    """Sum of squares utility function.

    Args:
            x: reward vector
    """
    return np.sum(np.pow(x, 2), dim=0)


def u_balance(x):
    """Product utility function.

    Args:
            x: reward vector
    """
    return np.prod(x, dim=0)


def mkdir_p(path):
    """Creates a folder at the provided path, used  for logging functionality.

    Args:
        path: string defining the location of the folder.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
