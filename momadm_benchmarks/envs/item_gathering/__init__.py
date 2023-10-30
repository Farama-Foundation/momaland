"""Item Gathering environment.

Johan Källström and Fredrik Heintz. 2019. Tunable Dynamics in Agent-Based Simulation using Multi-Objective
Reinforcement Learning. Presented at the Adaptive and Learning Agents Workshop at AAMAS 2019.
https://liu.diva-portal.org/smash/record.jsf?pid=diva2%3A1362933&dswid=9018

Notes:
    - In contrast to the original environment, the observation space is a 2D array of integers, i.e.,
    the map of the environment, where each integer represents either agents (1 for the agent receiving the observation,
     2 for the other agents) or items (3, 4, etc., depending on the number of items).
    - The number of agents and items is configurable, by providing an initial map.
    - If no initial map is provided, the environment uses a default map
"""
