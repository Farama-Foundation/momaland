[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
![tests](https://github.com/rradules/momaland/workflows/Python%20tests/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- start elevator-pitch -->
MOMAland is an open source Python library for developing and comparing multi-objective multi-agent reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API. Essentially, the environments follow the standard [PettingZoo APIs](https://github.com/Farama-Foundation/PettingZoo), but return vectorized rewards as numpy arrays instead of scalar values.

The documentation website is at TODO, and we have a public discord server (which we also use to coordinate development work) that you can join [here](https://discord.gg/bnJ6kubTg6).
<!-- end elevator-pitch -->

## Environments
MOMAland includes environments taken from the MOMARL literature, as well as multi-objective version of classical environments, such as SISL or Butterfly.
The full list of environments is available at TODO.

## Installation
<!-- start install -->
To install MOMAland, use:
```bash
pip install momaland
```
This does not include dependencies for all components of MOMAland (not everything is required for the basic usage, and some can be problematic to install on certain systems).
- `pip install "momaland[testing]"` to install dependencies for API testing.
- `pip install "momaland[learning]"` to install dependencies for the supplied learning algorithms.
- `pip install "momaland[all]"` for all dependencies for all components.
<!-- end install -->

## API
<!-- start snippet-usage -->
Similar to [PettingZoo](https://pettingzoo.farama.org), the MOMAland API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple - here's an example using the "momultiwalker_v0" environment:

```python
from momaland.envs.momultiwalker import momultiwalker_v0 as _env
import numpy as np

# .env() function will return an AEC environment, as per PZ standard
env = _env.env(render_mode="human")

# optionally, you can scalarize the reward with weights
env = momaland.LinearReward(env, weight=np.array([0.6, 0.2, 0.2]))

env.reset(seed=42)
for agent in env.agent_iter():
    # vec_reward is a numpy array
    observation, vec_reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()
```

For details on multi-objective multi-agent RL definitions, see [Multi-Objective Multi-Agent Decision Making: A Utility-based Analysis and Survey](https://arxiv.org/abs/1909.02964).

You can also check more examples in this collab notebook! TODO
<!-- end snippet-usage -->

## Environment Versioning
MOMAland keeps strict versioning for reproducibility reasons. All environments end in a suffix like "_v0".  When changes are made to environments that might impact learning results, the number is increased by one to prevent potential confusion.

## Development Roadmap
We have a roadmap for future development available here: TODO.

## Project Maintainers
Project Managers:  TODO

Maintenance for this project is also contributed by the broader Farama team: [farama.org/team](https://farama.org/team).

## Citing
<!-- start citation -->
If you use this repository in your research, please cite:
```bibtex
@inproceedings{TODO}
```
<!-- end citation -->

## Development
### Setup pre-commit
Clone the repo and run `pre-commit install` to setup the pre-commit hooks.
