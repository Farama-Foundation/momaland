---
title: Making a Custom Environment
firstpage:
---
## Setup pre-commit hooks
`pre-commit install` to initialize pre-commit hooks. `pip install pre-commit` if you don't have the package.

## Making a Custom Environment
1. Create a new environment package in `momaland/envs/`
2. Create a new environment class in `momaland/envs/<env_name>/<env_name>.py`, this class should extend `MOParallelEnv` or `MOAECEnv`. Override the PettingZoo methods (see their [documentation](https://pettingzoo.farama.org/api/aec/)). Additionally, you should define a member `self.reward_spaces` that is a dictionary of space specifying the shape of the reward vector of each agent, as well as a method `reward_space(self, agent) -> Space` that returns the reward space of a given agent.
3. Define the factory functions to create your class: `parallel_env` returns a parallel version of the env, `env` returns an AEC version, and `raw_env` that is the pure class constructor (it is not used in practice).
- (!) Use the conversions that are defined inside our repository, e.g. `mo_parallel_to_aec` instead of `parallel_to_aec` from PZ.
4. Add a versioned constructor of your env in the directory which exports the factory functions (see `mobeach_v0.py` for an example).
5. Add your environment to the tests in `utils/all_modules.py`
6. Run `pytest` to check that everything works
