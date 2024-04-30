"""Plot the Pareto front for the BPD problem."""

import argparse
from distutils.util import strtobool

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


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
    parser.add_argument('--show-lines', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="run with random actions")
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

    colors = [
        "#5CB5FF",
        "#D55E00",
        "#009E73",
        "#e6194b",
    ]

    linestyle = "None" if not args.show_lines else "-"

    # Plot the data
    ax1 = plt.plot(df_pf["Capacity"], df_pf["Mixture"], marker=".", linestyle=linestyle, color=colors[0])
    ax2 = plt.plot(df_runs_g["Capacity"], df_runs_g["Mixture"], marker=".", linestyle=linestyle, color=colors[1])
    ax3 = plt.plot(df_runs_l["Capacity"], df_runs_l["Mixture"], marker=".", linestyle=linestyle, color=colors[2])
    ax4 = plt.plot(df_runs_random["Capacity"], df_runs_random["Mixture"], marker=".", linestyle=linestyle, color=colors[3])

    # Set the title of the plot
    plt.title(f"{args.num_agents} Agents")
    # Set the labels for the x and y axes
    plt.xlabel("Capacity")
    plt.ylabel("Mixture")

    plt.grid(alpha=0.3)
    plt.legend(
        ["True PF", "Global", "Local", "Random"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.9),
        bbox_transform=plt.gcf().transFigure,
        ncol=4,
        fontsize="16",
    )
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"momaland/learning/iql/results/PF_{args.num_agents}.pdf", bbox_inches="tight")
    plt.show(bbox_inches="tight")
