---
autogenerated:
title: MO-RouteChoice
firstpage:
---

# MO-RouteChoice

|   |                                   |
|---|-----------------------------------|
| Agents names | `agent_i for i in [0, 4199]`      |
| Action Space | Discrete(3)                       |
| Observation Space | Box(0.0, 4200.0, (1,), float32)   |
| Reward Space | Box(-3.0, 0.0, (2,), float32)     |
| Import | `momaland.envs.mo_routechoice_v0` |

Environment for MO-RouteChoice problem.

The init method takes in environment arguments and should define the following attributes:
- possible_agents
- action_spaces
- observation_spaces
These attributes should not be changed after initialization.