---
title: "AEC"
---

# AEC
## Usage
AEC environments can be interacted with as follows:

```python
from momaland.envs.momultiwalker import momultiwalker_v0 as _env

# .env() function will return an AEC environment, as per PZ standard
env = _env.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    # reward is a numpy array
    observation, vec_reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()
```

MOMAland environments extend the base `MOAEC` class, as opposed to PettingZoo's base `AEC` class. `MOAEC` extends `AEC` and has a `reward_space` member.

For more detailed documentation on the AEC API, [see the PettingZoo documentation](https://pettingzoo.farama.org/api/aec/).
