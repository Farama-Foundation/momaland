"""Derivation of the PF in the Beach Domain."""
from itertools import product

import numpy as np
import pandas as pd

from momaland.envs.beach_domain.beach_domain import (
    _local_capacity_reward,
    _local_mixture_reward,
)
from momaland.learning.utils import mkdir_p


def fast_p_prune(candidates):
    """A batched and fast version of the Pareto coverage set algorithm.

    From ramo: https://github.com/wilrop/ramo
    Args:
        candidates (ndarray): A numpy array of vectors.

    Returns:
        ndarray: A Pareto coverage set.
    """
    candidates = np.unique(candidates, axis=0)
    if len(candidates) == 1:
        return candidates

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == 1
    c2 = np.any(~res_g, axis=-1)
    return list(candidates[np.logical_and(c1, c2)])


"""
# Sanity check
def generate_distributions2(num_agents, capacity):
    distr = []
    for i in range(capacity + 1):
        for j in range(capacity + 1):
            for k in range(capacity + 1):
                for l in range(capacity + 1):
                    m = num_agents - i - j - k - l
                    if i + j + k + l + m == num_agents:
                        distr.append([i, j, k, l, m])
    return distr
"""


def generate_distributions(num_agents, num_sections, capacity):
    """Generate all possible distributions of agents into sections.

    Takes advantage of the knowledge that no solution will be optimal if we have more than
    'capacity' agents on the first 'num_sections - 1'
    """
    distr = []
    prod = product([i for i in range(0, capacity + 1)], repeat=num_sections - 1)
    for i in list(prod):
        last = num_agents - sum(i)
        if last >= 0:
            distr.append(np.append(list(i), last))
    return distr


def prune_data(data, pf):
    """Keep only the data and configurations that are part of the PF."""
    final = []
    for d in data:
        for p in pf:
            if d[0] == p[0] and d[1] == p[1]:
                final.append(d)
    return final


def get_PF(distributions_a, distributions_b, capacity, total_agents, num_sections, folder):
    """Get the PF for both local and global reward structures for  the Beach Domain and save it to a csv."""
    solutions_l = []
    solutions_g = []
    data_l = []
    data_g = []
    iter = 0
    for distr_a in distributions_a:
        if iter % 100 == 0:
            print(iter)
        iter += 1
        for distr_b in distributions_b:
            all_rew = []
            cons = []
            for i in range(num_sections):
                el = [distr_a[i], distr_b[i]]
                cons.append(el)
                local_cap = _local_capacity_reward(capacity, sum(el))
                local_mix = _local_mixture_reward(el)
                all_rew.append([local_cap, local_mix])
            # make sure distribution is correct
            assert np.sum(cons) == total_agents
            # compute the average local reward over all sections for all objectives
            avg_r = np.average(all_rew, axis=0)
            # compute the global reward over all sections for all objectives
            sum_r = np.sum(all_rew, axis=0)
            solutions_l.append(avg_r)
            solutions_g.append(sum_r)
            data_l.append([avg_r[0], avg_r[1], cons])
            data_g.append([sum_r[0], sum_r[1], cons])

        # intermediate pruning and saving steps
        solutions_l = fast_p_prune(solutions_l)
        solutions_g = fast_p_prune(solutions_g)

        df = pd.DataFrame(solutions_l, columns=["Capacity", "Mixture"])
        df.to_csv(f"{folder}/pf_bpd_{total_agents}_L.csv", index=False)
        df = pd.DataFrame(solutions_g, columns=["Capacity", "Mixture"])
        df.to_csv(f"{folder}/pf_bpd_{total_agents}_G.csv", index=False)

    # final pruning and saving steps for the local and global PF
    pf = fast_p_prune(np.array(solutions_l))
    df = pd.DataFrame(pf, columns=["Capacity", "Mixture"])
    df.to_csv(f"{folder}/pf_bpd_{total_agents}_L.csv", index=False)
    # keep only the data and configurations that are part of the PF
    final = prune_data(data_l, pf)
    df = pd.DataFrame(final, columns=["Capacity", "Mixture", "Distribution"])
    df.to_csv(f"{folder}/data_bpd_{total_agents}_L.csv", index=False)

    pf = fast_p_prune(np.array(solutions_g))
    df = pd.DataFrame(pf, columns=["Capacity", "Mixture"])
    df.to_csv(f"{folder}/pf_bpd_{total_agents}_G.csv", index=False)
    # keep only the data and configurations that are part of the PF
    final = prune_data(data_g, pf)
    df = pd.DataFrame(final, columns=["Capacity", "Mixture", "Distribution"])
    df.to_csv(f"{folder}/data_bpd_{total_agents}_G.csv", index=False)


if __name__ == "__main__":
    """
    sec = [[10, 7], [8, 8], [17, 0]]
    sec = [[5, 5], [5, 5], [25, 5]]
    all_rew = []
    for el in sec:
        local_cap = _local_capacity_reward(capacity, sum(el))
        local_mix = _local_mixture_reward(el)
        all_rew.append([local_cap, local_mix])
        print(local_cap, local_mix)
    print(np.average(all_rew, axis=0), np.sum(all_rew, axis=0))
    """

    total_agents = 50

    if total_agents == 50:
        num_sections = 3
        capacity = 10
        type_distribution = [35, 15]
    elif total_agents == 100:
        num_sections = 5
        capacity = 10
        type_distribution = [70, 30]

    folder = "results/pf"
    mkdir_p(folder)

    # generate all combinations of distributions for each type
    distributions_a = generate_distributions(type_distribution[0], num_sections, capacity)
    distributions_b = generate_distributions(type_distribution[1], num_sections, capacity)
    print(len(distributions_a), len(distributions_b))

    get_PF(distributions_a, distributions_b, capacity, total_agents, num_sections, folder)
