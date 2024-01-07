"""Wrappers for training.

Parallel only.

TODO AEC.
"""

import os
from typing import List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from distrax import MultivariateNormalDiag


def _ma_get_pi(actor_module, params, obs: jnp.ndarray, agents):
    """Gets the actions for all agents at once. This is done with a for loop because distrax does not like vmapping."""
    return [actor_module.apply(params, obs[i]) for i in range(agents)]


def _ma_sample_and_log_prob_from_pi(pi: List[MultivariateNormalDiag], agents: int, key: chex.PRNGKey):
    """Samples actions for all agents in all the envs at once. This is done with a for loop because distrax does not like vmapping.

    Args:
        pi (List[MultivariateNormalDiag]): List of distrax distributions for agent actions (batched over envs).
        key (chex.PRNGKey): PRNGKey to use for sampling: size should be (num_agents, 2).
    """
    return [pi[i].sample_and_log_prob(seed=key[i]) for i in range(agents)]


def eval_mo(
    actor_module,
    actor_state,
    env,
    w: Optional[np.ndarray] = None,
    scalarization=np.dot,
    render: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates one episode of the agent in the environment.

    Args:
        actor_module: the initialized actor module
        actor_state: Agent
        env: MOMAland environment with LinearReward wrapper
        scalarization: scalarization function, taking weights and reward as parameters
        w (np.ndarray): Weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return
    """
    key = jax.random.PRNGKey(seed=42)
    key, subkey = jax.random.split(key)
    obs, _ = env.reset()
    np_obs = np.stack([obs[agent] for agent in env.possible_agents])
    done = False
    vec_return, disc_vec_return = np.zeros_like(w), np.zeros_like(w)
    gamma = 1.0
    pi = _ma_get_pi(actor_module, actor_state.params, jnp.array(np_obs), len(env.possible_agents))
    action_keys = jax.random.split(subkey, len(env.possible_agents))
    while not done:
        if render:
            env.render()

        # for each agent sample an action
        actions, log_probs = zip(*_ma_sample_and_log_prob_from_pi(pi, len(env.possible_agents), action_keys))
        actions_dict = dict()
        for i, agent in enumerate(env.possible_agents):
            actions_dict[agent] = np.array(actions[i])
        obs, rew, term, trun, info = env.step(actions_dict)
        done = term or trun

        # vec_return += rew
        # disc_vec_return += gamma * rew

        for i, _ in enumerate(vec_return):
            acc = 0
            for j, _ in enumerate(rew.values()):
                acc += list(rew.values())[j][i]
            vec_return[i] = acc / len(w)
        disc_vec_return = gamma * vec_return
        gamma *= 0.99

    if w is None:
        scalarized_return = scalarization(vec_return)
        scalarized_discounted_return = scalarization(disc_vec_return)
    else:
        scalarized_return = scalarization(w, vec_return)
        scalarized_discounted_return = scalarization(w, disc_vec_return)

    return (
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        disc_vec_return,
    )


def policy_evaluation_mo(
    actor_module, actor_state, env, w: np.ndarray, rep: int = 5
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates the value of a policy by running the policy for multiple episodes. Returns the average returns.

    Args:
        actor_module: the initialized actor module
        actor_state: Agent
        env: MOMAland environment
        w (np.ndarray): Weight vector
        rep (int, optional): Number of episodes for averaging. Defaults to 5.

    Returns:
        (float, float, np.ndarray, np.ndarray): Avg scalarized return, Avg scalarized discounted return, Avg vectorized return, Avg vectorized discounted return
    """
    evals = [eval_mo(actor_module, actor_state, env, w) for _ in range(rep)]
    avg_scalarized_return = np.mean([eval[0] for eval in evals])
    avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals])
    avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
    avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)

    return (
        avg_scalarized_return,
        avg_scalarized_discounted_return,
        avg_vec_return,
        avg_disc_vec_return,
    )


def save_results(returns, exp_name, seed):
    """Saves the results of an experiment to a csv file.

    Args:
        returns: a list of triples (timesteps, time, episodic_return)
        exp_name: experiment name
        seed: seed of the experiment
    """
    if not os.path.exists("results"):
        os.makedirs("results")
    filename = f"results/{exp_name}_{seed}.csv"
    print(f"Saving results to {filename}")
    df = pd.DataFrame(returns)
    df.columns = ["Total timesteps", "Time", "Episodic return"]
    df.to_csv(filename, index=False)
