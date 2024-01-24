---
title: All Environments
firstpage:
---

# All Environments
```{toctree}
:hidden:
:glob:
:caption: All Environments

./*
```

MOMAland includes environments taken from the MO/MARL literature, as well as multi-objective versions of environments from PettingZoo.

| Env                                                                                                                                                                                     | Cooperative/Adversarial | Obs/Action spaces       | Objectives                                 | Description                                                                                                                                                |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|-------------------------|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`catch-v0`](https://momaland.farama.org/environments/catch/) <br><img src="https://momaland.farama.org/_static/gifs/catch.gif" width="400px">                               | Cooperative             | Continuous / Continuous | `[distance_target, distance_other_drones]` | Agents must corner and catch a target drone while maintaining distance between themselves.                                                                 |
| [`escort-v0`](https://momaland.farama.org/environments/escort/) <br><img src="https://momaland.farama.org/_static/gifs/escort.gif" width="400px">                            | Cooperative                        | Continuous / Continuous | `[distance_target, distance_other_drones]` | Agents must circle around a mobile target drone and escort it to its destination without breaking formation while maintaining distance between themselves. |
| [`mobeach-v0`](https://momaland.farama.org/environments/mobeach/) <br><img src="https://momaland.farama.org/_static/gifs/mobeach.gif" width="400px">                         |                         | Continuous / Discrete   | `[TODO]`                                   | TODO                                                                                                                                                       |
| [`mocongestion-v0`](https://momaland.farama.org/environments/mocongestion/) <br><img src="https://momaland.farama.org/_static/gifs/mocongestion.gif" width="400px">          |                         | Continuous / Discrete   | `[TODO]`                                   | TODO                                                                                                                                                       |
| [`moitemgathering-v0`](https://momaland.farama.org/environments/moitemgathering/) <br><img src="https://momaland.farama.org/_static/gifs/moitemgathering.gif" width="400px"> | Adversarial                        | Discrete / Discrete     | `[TODO]`                                   | TODO                                                                                                                                                       |
| [`momultiwalker-v0`](https://momaland.farama.org/environments/momultiwalker/) <br><img src="https://momaland.farama.org/_static/gifs/momultiwalker.gif" width="400px">       | Cooperative                        | Continuous / Continuous | `[TODO]`                                   | TODO                                                                                                                                                       |
| [`mopistonball-v0`](https://momaland.farama.org/environments/mopistonball/) <br><img src="https://momaland.farama.org/_static/gifs/mopistonball.gif" width="400px">          | Cooperative                        | TODO / TODO             | `[TODO]`                                   | TODO                                                                                                                                                       |
| [`surround-v0`](https://momaland.farama.org/environments/surround/) <br><img src="https://momaland.farama.org/_static/gifs/surround.gif" width="400px">                      | Cooperative                        | Continuous / Continuous | `[distance_target, distance_other_drones]` | Agents must surround a fixed target point while maintaining distance between themselves.                                                                   |
