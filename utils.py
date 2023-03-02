import os
import errno
import numpy as np


def u_imbalance(x):
    return np.sum(np.pow(x, 2), dim=0)


def u_balance(x):
    return np.prod(x, dim=0)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
