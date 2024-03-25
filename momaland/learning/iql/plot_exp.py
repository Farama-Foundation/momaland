"""Plot the scalarized reward of the best runs for the different reward schemes."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    """Argument parsing for pareto front plot."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-agents', type=int, default=50, help="Number of agents")
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()

    # Read the CSV file with best runs
    df_runs_g = pd.read_csv(f"momaland/learning/iql/results/runs/BPD_{args.num_agents}_global.csv")
    df_runs_l = pd.read_csv(f"momaland/learning/iql/results/runs/BPD_{args.num_agents}_local.csv")
    df_random = pd.read_csv(f"momaland/learning/iql/results/runs/BPD_{args.num_agents}_random.csv")

    # Add a column to the dataframes to distinguish between the different reward schemes
    df_runs_l["Reward"] = "Local"
    df_runs_g["Reward"] = "Global"
    df_random["Reward"] = "Random"

    # Concatenate the dataframes
    df_total = pd.concat([df_runs_l, df_runs_g, df_random])

    # Plot the data
    sns.lineplot(data=df_total, x="episode", y="scal_rew", hue="Reward")
    plt.title(f"Num Agents {args.num_agents} (BPD)")
    plt.xlabel("Episode")
    plt.ylabel("Scalarized Reward")

    # Display the plot
    plt.show()
