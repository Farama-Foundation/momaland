![tests](https://github.com/rradules/momaland/workflows/Python%20tests/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<!-- start elevator-pitch -->

MOMAland is an open source Python library for developing and comparing multi-objective multi-agent reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API. Essentially, the environments follow the standard [PettingZoo APIs](https://github.com/Farama-Foundation/PettingZoo), but return vectorized rewards as numpy arrays instead of scalar values.

The documentation website is at TODO, and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6.

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

This does not include dependencies for all families of environments (some can be problematic to install on certain systems). You can install these dependencies for one family like `pip install "momaland"` or use `pip install "momaland[all]"` to install all dependencies.

<!-- end install -->

## API

<!-- start snippet-usage -->

As for PettingZoo, the MOMAland API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple - here's an example using the "mosurround_v0" environment:

```python
import momaland
import numpy as np

# It follows the original PettingZoo APIs ...
env = momaland.envs.crazyrl.surround.surround_v0.parallel_env()

obs, info = env.reset()
# but vector_reward is a numpy array!
actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
next_obs, vector_rewards, terminated, truncated, info = env.step(actions)

# Optionally, you can scalarize the reward function with the LinearReward wrapper to fall back to the original PZ API
env = momaland.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))
```
For details on multi-objective multi-agent RL definitions, see [Multi-Objective Multi-Agent Decision Making: A Utility-based Analysis and Survey](https://arxiv.org/abs/1909.02964).

You can also check more examples in this colab notebook! TODO

<!-- end snippet-usage -->


## Environment Versioning

MOMAland keeps strict versioning for reproducibility reasons. All environments end in a suffix like "-v0".  When changes are made to environments that might impact learning results, the number is increased by one to prevent potential confusion.

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

### New environment steps
1. Create a new environment package in `momaland/envs/`
2. Create a new environment class in `momaland/envs/<env_name>/<env_name>.py`, this class should extend `MOParallelEnv` or `MOAECEnv`. Override the PettingZoo methods (see their [documentation](https://pettingzoo.farama.org/api/aec/)). Additionally, you should define a member `self.reward_spaces` that is a dictionary of space specifying the shape of the reward vector of each agent, as well as a method `reward_space(self, agent) -> Space` that returns the reward space of a given agent.
3. Define the factory functions to create your class: `parallel_env` returns a parallel version of the env, `env` returns an AEC version, and `raw_env` that is the pure class constructor (it is not used in practice). (!) use the conversions that are defined inside our repository, e.g. `mo_parallel_to_aec` instead of `parallel_to_aec` from PZ.
4. (!) do not use `OrderEnforcingWrapper`, it prevents from accessing the `reward_space` of the env :-(;
5. Add a versioned constructor of your env in the directory which exports the factory functions (see `mobeach_v0.py` for an example).
6. Add your environment to the tests in `utils/all_modules.py`
7. Run `pytest` to check that everything works
