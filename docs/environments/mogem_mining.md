---
autogenerated:
title: MO-GemMining
firstpage:
---

# MO-GemMining

|   |   |
|---|---|
| Agents names | `agent_i for i in [0, 19]` |
| Action Space | ['0: Discrete(4)', '1: Discrete(3, start=1)', '2: Discrete(4, start=2)', '3: Discrete(2, start=3)', '4: Discrete(3, start=4)', '5: Discrete(3, start=5)', '6: Discrete(4, start=6)', '7: Discrete(3, start=7)', '8: Discrete(4, start=8)', '9: Discrete(3, start=9)', '10: Discrete(4, start=10)', '11: Discrete(2, start=11)', '12: Discrete(4, start=12)', '13: Discrete(4, start=13)', '14: Discrete(2, start=14)', '15: Discrete(4, start=15)', '16: Discrete(4, start=16)', '17: Discrete(3, start=17)', '18: Discrete(4, start=18)', '19: Discrete(3, start=19)'] |
| Observation Space | Box(0.0, 20.0, (1,), float32) |
| Reward Space | Box(0.0, 23.0, (2,), float32) |
| Import | `momaland.envs.mogem_mining_v0` |

Environment for MO-GemMining domain.

The init method takes in environment arguments and should define the following attributes:
- possible_agents
- action_spaces
- observation_spaces
These attributes should not be changed after initialization.
