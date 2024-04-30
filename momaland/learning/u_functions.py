"""Utility functions for MOMAland."""

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
