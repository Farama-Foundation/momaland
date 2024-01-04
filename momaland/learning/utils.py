"""Helper functions for MOMAland training and logging."""

import errno
import os

import pandas as pd


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


def save_results(returns, exp_name, seed):
    """Saves the results of an experiment to a csv file.

    Args:
        returns: a list of triples (timesteps, time, episodic_return)
        exp_name: experiment name
        seed: seed of the experiment
    """
    mkdir_p("results")
    filename = f"results/{exp_name}_{seed}.csv"
    print(f"Saving results to {filename}")
    df = pd.DataFrame(returns)
    df.columns = ["Total timesteps", "Time", "Episodic return"]
    df.to_csv(filename, index=False)


def map_actions(actions, num_actions):
    """Map a list of actions to a single number to create a Discrete action space for MORL baselines."""
    return sum([actions[i] * num_actions**i for i in range(len(actions))])


def remap_actions(action, num_agents, num_actions):
    """Remap a single number to a list of actions."""
    actions = []
    for i in range(num_agents):
        actions.append(action % num_actions)
        action = action // num_actions
    return actions
