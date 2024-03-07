"""Train IQL on the Beach Problem Domain, under linear scalarization, and save the results to a csv file."""
import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import pandas as pd

from momaland.learning.discrete.iql import train


def parse_args():
    """Argument parsing for hyperparameter optimization."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument('--seed', type=int, default=1,
                        help="the seed of the experiment")
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="run in debug mode")

    # Algorithm specific parameters
    parser.add_argument('--num-iterations', type=int, default=10000,
                        help="the number of training iterations")
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="the learning rate")
    parser.add_argument('--alpha-decay', type=float, default=1,
                        help="the learning rate decay")
    parser.add_argument('--alpha-min', type=float, default=0.1,
                        help="the minimum learning rate")
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help="the exploration rate")
    parser.add_argument('--epsilon-decay', type=float, default=0.9999,
                        help="the exploration rate decay")
    parser.add_argument('--epsilon-min', type=float, default=0.0,
                        help="the minimum exploration rate")
    parser.add_argument('--gamma', type=float, default=0.9,
                        help="the discount rate")

    args = parser.parse_args()
    args.time = time.time()

    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()

    solutions = [[], []]
    for w1 in np.round(np.linspace(0, 1, 21), 2):
        w2 = round(1 - w1, 2)
        print(w1, w2)
        metrics = train(args, np.array([w1, w2]))
        er = metrics["metrics"]["returned_episode_returns"][-1][2]
        solutions.append([w1, w2, er["avg_obj1"], er["avg_obj2"]])
        print(f"metrics: {metrics['metrics']['returned_episode_returns'][-1]}")

    # Save the solutions to csv
    df = pd.DataFrame(solutions, columns=["w1", "w2", "obj1", "obj2"])

    df.to_csv(f"results/BPD_G_{args.exp_name}_{args.seed}.csv", index=False)
