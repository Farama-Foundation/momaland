---
title: "Parallel"
---

# Parallel
## Usage
Parallel environments can be interacted with as follows:

```python
from momaland.envs.momultiwalker import momultiwalker_v0 as _env

# .parallel_env() function will return a Parallel environment, as per PZ standard
parallel_env = _env.parallel_env(render_mode="human")
observations, infos = parallel_env.reset(seed=42)

while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}

    # reward is a numpy array
    observations, vec_rewards, terminations, truncations, infos = parallel_env.step(actions)
parallel_env.close()
```

MOMAland environments extend the base `MOParallel` class, as opposed to PettingZoo's base `Parallel` class. `MOParallel` extends `Parallel` and has a `reward_space` member.

For more detailed documentation on the Parallel API, [see the PettingZoo documentation](https://pettingzoo.farama.org/api/parallel/).
