---
title: Environments
firstpage:
---

# Available environments

```{toctree}
:hidden:
:glob:
:caption: MOMAland environments

./*
```

MOMAland includes environments taken from the MO/MARL literature, as well as multi-objective versions of environments from PettingZoo.

| Env                                                                                                                                                                                                                                                                | Obs/Action spaces                   | Objectives                                                    | Description                                                                                                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`catch-v0`](https://rradules.github.io/momaland/environments/catch/) <br><img src="https://rradules.github.io/momaland/_images/catch.gif" width="400px">                            | Continuous / Continuous                 | `[distance_target, distance_other_drones]`                                    | Agents must corner and catch a target drone while maintaining distance between themselves.                                                                               |
| [`escort-v0`](https://rradules.github.io/momaland/environments/escort/) <br><img src="https://rradules.github.io/momaland/_images/escort.gif" width="400px">                            | Continuous / Continuous                 | `[distance_target, distance_other_drones]`                                    | Agents must circle around a mobile target drone and escort it to its destination without breaking formation while maintaining distance between themselves.                                                                               |
| [`mobeach-v0`](https://rradules.github.io/momaland/environments/mobeach/) <br><img src="https://rradules.github.io/momaland/_images/mobeach.gif" width="400px">                            | Continuous / Discrete                 | `[TODO]`                                    | TODO                                                                               |
| [`mocongestion-v0`](https://rradules.github.io/momaland/environments/mocongestion/) <br><img src="https://rradules.github.io/momaland/_images/mocongestion.gif" width="400px">                            | Continuous / Discrete                 | `[TODO]`                                    | TODO                                                                               |
| [`moitemgathering-v0`](https://rradules.github.io/momaland/environments/moitemgathering/) <br><img src="https://rradules.github.io/momaland/_images/moitemgathering.gif" width="400px">                            | Discrete / Discrete                 | `[TODO]`                                    | TODO                                                                               |
| [`momultiwalker-v0`](https://rradules.github.io/momaland/environments/momultiwalker/) <br><img src="https://rradules.github.io/momaland/_images/momultiwalker.gif" width="400px">                            | Continuous / Continuous                 | `[TODO]`                                    | TODO                                                                               |
| [`mopistonball-v0`](https://rradules.github.io/momaland/environments/mopistonball/) <br><img src="https://rradules.github.io/momaland/_images/mopistonball.gif" width="400px">                            | TODO / TODO                 | `[TODO]`                                    | TODO                                                                               |
| [`surround-v0`](https://rradules.github.io/momaland/environments/surround/) <br><img src="https://rradules.github.io/momaland/_images/surround.gif" width="400px">                            | Continuous / Continuous                 | `[distance_target, distance_other_drones]`                                    | Agents must surround a fixed target point while maintaining distance between themselves.                                                                               |
