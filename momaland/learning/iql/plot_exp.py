"""Plot the scalarized reward of the best runs for the different reward schemes."""

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


mpl.rcParams["pdf.fonttype"] = 42  # use true-type
mpl.rcParams["ps.fonttype"] = 42  # use true-type
mpl.rcParams["font.size"] = 18
mpl.rcParams["lines.linewidth"] = 2.2
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb,underscore}"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["axes.titlesize"] = 18
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 16


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
    df_runs_l["Reward Scheme"] = "Local"
    df_runs_g["Reward Scheme"] = "Global"
    df_random["Reward Scheme"] = "Random"

    # Concatenate the dataframes
    df_total = pd.concat([df_runs_l, df_runs_g, df_random])

    # Plot the data
    colors = [
        "#5CB5FF",
        "#D55E00",
        "#009E73",
        # "#e6194b",
    ]
    ax = sns.lineplot(data=df_total, x="episode", y="scal_rew", hue="Reward Scheme", palette=colors)
    plt.title(f"{args.num_agents} Agents")
    plt.xlabel("Episodes")
    plt.ylabel("Scalarized Reward")
    plt.ylim(0.4, 0.67)
    plt.grid(alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    plt.legend(
        h, l, loc="lower center", bbox_to_anchor=(0.5, 0.9), bbox_transform=plt.gcf().transFigure, ncol=3, fontsize="16"
    )
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"momaland/learning/iql/results/BPD_{args.num_agents}.pdf", bbox_inches="tight")
    plt.show(bbox_inches="tight")
