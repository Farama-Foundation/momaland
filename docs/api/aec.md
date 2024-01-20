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

## MOAECEnv

```{eval-rst}
.. currentmodule:: momaland.utils.env

.. autoclass:: MOAECEnv

```

## Attributes
```{eval-rst}
.. autoattribute:: MOAECEnv.agents

    A list of the names of all current agents, typically integers. These may be changed as an environment progresses (i.e. agents can be added or removed).

    :type: List[AgentID]

.. autoattribute:: MOAECEnv.num_agents

    The length of the agents list.

.. autoattribute:: MOAECEnv.possible_agents

    A list of all possible_agents the environment could generate. Equivalent to the list of agents in the observation and action spaces. This cannot be changed through play or resetting.

    :type: List[AgentID]

.. autoattribute:: MOAECEnv.max_num_agents

    The length of the possible_agents list.

.. autoattribute:: MOAECEnv.agent_selection

    An attribute of the environment corresponding to the currently selected agent that an action can be taken for.

    :type: AgentID

.. autoattribute:: MOAECEnv.terminations

.. autoattribute:: MOAECEnv.truncations

.. autoattribute:: MOAECEnv.rewards

    A dict of the rewards of every current agent at the time called, keyed by name. Rewards the instantaneous reward generated after the last step. Note that agents can be added or removed from this attribute. `last()` does not directly access this attribute, rather the returned reward is stored in an internal variable. The rewards structure looks like::

    {0:[first agent reward], 1:[second agent reward] ... n-1:[nth agent reward]}

    :type: Dict[AgentID, float]

.. autoattribute:: MOAECEnv.infos

    A dict of info for each current agent, keyed by name. Each agent's info is also a dict. Note that agents can be added or removed from this attribute. `last()` accesses this attribute. The returned dict looks like::

        infos = {0:[first agent info], 1:[second agent info] ... n-1:[nth agent info]}

    :type: Dict[AgentID, Dict[str, Any]]

.. autoattribute:: MOAECEnv.observation_spaces

    A dict of the observation spaces of every agent, keyed by name. This cannot be changed through play or resetting.

    :type: Dict[AgentID, gymnasium.spaces.Space]

.. autoattribute:: MOAECEnv.action_spaces

    A dict of the action spaces of every agent, keyed by name. This cannot be changed through play or resetting.

    :type: Dict[AgentID, gymnasium.spaces.Space]

.. autoattribute:: MOAECEnv.reward_spaces

    A dict of the reward spaces of every agent, keyed by name. This cannot be changed through play or resetting.

    :type: Dict[AgentID, gymnasium.spaces.Space]
```

## Methods
```{eval-rst}
.. automethod:: MOAECEnv.step
.. automethod:: MOAECEnv.reset
.. automethod:: MOAECEnv.observe
.. automethod:: MOAECEnv.render
.. automethod:: MOAECEnv.close
.. automethod:: MOAECEnv.observation_space
.. automethod:: MOAECEnv.action_space
.. automethod:: MOAECEnv.reward_space
```
