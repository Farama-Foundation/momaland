---
title: Cooperative Multi-Agent Environments
firstpage:
---

## Cooperative
```{toctree}
:hidden:
:glob:
:caption: Cooperative Environments

./cooperative/*
```

MOMAland includes environments taken from the MO/MARL literature, as well as multi-objective versions of environments from PettingZoo.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`catch-v0`](https://rradules.github.io/momaland/environments/cooperative/catch/) <br><img src="https://rradules.github.io/momaland/_images/catch.gif" width="400px">                            | Continuous / Continuous                 | `[distance_target, distance_other_drones]`                                    | Agents must corner and catch a target drone while maintaining distance between themselves.                                                                               |
| [`escort-v0`](https://rradules.github.io/momaland/environments/cooperative/escort/) <br><img src="https://rradules.github.io/momaland/_images/escort.gif" width="400px">                            | Continuous / Continuous                 | `[distance_target, distance_other_drones]`                                    | Agents must circle around a mobile target drone and escort it to its destination without breaking formation while maintaining distance between themselves.    |
| [`surround-v0`](https://rradules.github.io/momaland/environments/cooperative/surround/) <br><img src="https://rradules.github.io/momaland/_images/surround.gif" width="400px">                            | Continuous / Continuous                 | `[distance_target, distance_other_drones]`                                    | Agents must surround a fixed target point while maintaining distance between themselves.                                                                               |
| [`mopistonball-v0`](https://rradules.github.io/momaland/environments/cooperative/mopistonball/) <br><img src="https://rradules.github.io/momaland/_images/mopistonball.gif" width="400px">                            | TODO / TODO                 | `[TODO]`                                    | TODO                                                                               |
| [`momultiwalker-v0`](https://rradules.github.io/momaland/environments/cooperative/momultiwalker/) <br><img src="https://rradules.github.io/momaland/_images/momultiwalker.gif" width="400px">                            | Continuous / Continuous                 | `[TODO]`                                    | TODO                                                                               |
