![tests](https://github.com/rradules/momadm-bechmarks/workflows/Python%20tests/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# momadm-bechmarks
Benchmarks for Multi-Objective Multi-Agent Decision Making


## Development

### Setup pre-commit
Clone the repo and run `pre-commit install` to setup the pre-commit hooks.

### New environment steps
1. Create a new environment package in `momadm_benchmarks/envs/`
2. Create a new environment class in `momadm_benchmarks/envs/<env_name>/<env_name>.py`, this class should extend `MOParallelEnv` or `MOAECEnv`. Override the PettingZoo methods (see their [documentation](https://pettingzoo.farama.org/api/aec/)). Additionally, you should define a member `self.reward_spaces` that is a dictionary of space specifying the shape of the reward vector of each agent, as well as a method `reward_space(self, agent) -> Space` that returns the reward space of a given agent.
3. Define the factory functions to create your class: `parallel_env` returns a parallel version of the env, `env` returns an AEC version, and `raw_env` that is the pure class constructor (it is not used in practice). (!) use the conversions that are defined inside our repository, e.g. `mo_parallel_to_aec` instead of `parallel_to_aec` from PZ.
4. (!) do not use `OrderEnforcingWrapper`, it prevents from accessing the `reward_space` of the env :-(;
5. Add a versioned constructor of your env in the directory which exports the factory functions (see `mobeach_v0.py` for an example).
6. Add your environment to the tests in `tests/all_modules.py`
7. Run `pytest` to check that everything works
