"""Utils for the learning module."""

import errno
import os
import subprocess

import numpy as np
import requests


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


def map_actions(actions, num_actions):
    """Map a list of actions to a single number to create a Discrete action space for MORL baselines."""
    return sum([actions[i] * num_actions**i for i in range(len(actions))])


def remap_actions(action, num_agents, num_actions):
    """Remap a single number to a list of actions."""
    return np.unravel_index(action, (num_actions,) * num_agents)


def autotag():
    """This adds a tag to the wandb run marking the commit number, allows versioning of experiments. From CleanRL's benchmark utility."""

    def _autotag() -> str:
        wandb_tag = ""
        print("autotag feature is enabled")
        try:
            git_tag = subprocess.check_output(["git", "describe", "--tags"]).decode("ascii").strip()
            wandb_tag = f"{git_tag}"
            print(f"identified git tag: {git_tag}")
        except subprocess.CalledProcessError:
            return wandb_tag

        git_commit = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"]).decode("ascii").strip()
        try:
            # try finding the pull request number on github
            prs = requests.get(f"https://api.github.com/search/issues?q=repo:Farama-Foundation/momaland+is:pr+{git_commit}")
            if prs.status_code == 200:
                prs = prs.json()
                if len(prs["items"]) > 0:
                    pr = prs["items"][0]
                    pr_number = pr["number"]
                    wandb_tag += f",pr-{pr_number}"
            print(f"identified github pull request: {pr_number}")
        except Exception as e:
            print(e)

        return wandb_tag

    if "WANDB_TAGS" in os.environ:
        raise ValueError(
            "WANDB_TAGS is already set. Please unset it before running this script or run the script with --auto-tag False"
        )
    wandb_tag = _autotag()
    if len(wandb_tag) > 0:
        os.environ["WANDB_TAGS"] = wandb_tag
