"""Plot the Pareto front for the BPD problem."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


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

    # Read the CSV file into a DataFrame
    df_pf = pd.read_csv(f"momaland/learning/iql/results/pf/pf_bpd_{args.num_agents}_G.csv")
    # Read the CSV file with best runs
    df_runs_g = pd.read_csv(f"momaland/learning/iql/results/nds/BPD_{args.num_agents}_global.csv")
    df_runs_l = pd.read_csv(f"momaland/learning/iql/results/nds/BPD_{args.num_agents}_local.csv")
    # Read the CSV file with random run
    df_runs_random = pd.read_csv(f"momaland/learning/iql/results/nds/BPD_{args.num_agents}_random.csv")

    # Plot the data
    plt.plot(df_pf["Capacity"], df_pf["Mixture"], marker="x")
    plt.plot(df_runs_g["Capacity"], df_runs_g["Mixture"], marker="x")
    plt.plot(df_runs_l["Capacity"], df_runs_l["Mixture"], marker="x")
    plt.plot(df_runs_random["Capacity"], df_runs_random["Mixture"], marker="x")
    plt.legend(["True PF", "Global", "Local", "Random"])

    # Set the title of the plot
    plt.title(f"Num Agents {args.num_agents} (BPD)")
    # Set the labels for the x and y axes
    plt.xlabel("Capacity")
    plt.ylabel("Mixture")

    # Display the plot
    plt.show()
