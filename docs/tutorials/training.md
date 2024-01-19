---
title: Training
---

# Training
There are learning algorithms present in the codebase for developers to get familiar with MOMA training. Various MOMA algorithms can be accessed [here](https://github.com/rradules/momaland/tree/main/momaland/learning).

Below an example of **MAPPO with OLS weight generation to train MO** can be be found: [source](https://github.com/rradules/momaland/blob/main/momaland/learning/continuous/cooperative_momappo.py)
```python
args = parse_args()
rng = jax.random.PRNGKey(args.seed)
np.random.seed(args.seed)

start_time = time.time()

# NN initialization and jit compiled functions
env: ParallelEnv = Catch.parallel_env()
eval_env: ParallelEnv = Catch.parallel_env()
eval_env = clip_actions_v0(eval_env)
eval_env = normalize_obs_v0(env, env_min=-1.0, env_max=1.0)
eval_env = agent_indicator_v0(eval_env)

env.reset()
eval_env.reset()
current_timestep = 0

single_action_space = env.action_space(env.possible_agents[0])
actor = Actor(single_action_space.shape[0], net_arch=args.actor_net_arch, activation=args.activation)
critic = Critic(net_arch=args.critic_net_arch, activation=args.activation)
vmapped_get_value = vmap(critic.apply, in_axes=(None, 0))
critic.apply = jax.jit(critic.apply)

if args.track:
    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{env.__class__.__name__}_{exp_name}_{args.seed}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        save_code=True,
    )

ols = LinearSupport(num_objectives=2, epsilon=0.0001, verbose=True)
value = []
while not ols.ended():
    w = ols.next_weight()
    out = train(args, env, w, rng)
    actor_state = out["runner_state"][0]
    _, disc_vec_return = policy_evaluation_mo(actor, actor_state, env=eval_env, num_obj=ols.num_objectives)
    value.append(disc_vec_return)
    ols.add_solution(value[-1], w)

env.close()
wandb.finish()
print(f"total time: {time.time() - start_time}")
```
