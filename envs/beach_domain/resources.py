import numpy as np


class BeachSection(object):
    def __init__(self, num, capacity):
        self.id = num
        self.capacity = capacity
        self.consumption = 0
        self.typeA = 0

    def add_load(self, group, types):
        if len(group) > 0:
            self.consumption = len(group)
            self.typeA = sum(types)/len(types)

    def get_load(self):
        return self.consumption

    def reset(self):
        self.consumption = 0
        self.typeA = 0

    def is_overused(self):
        return self.capacity < self.consumption

    def local_reward(self):
        lr_capacity = self.consumption * np.exp(-self.consumption / self.capacity)
        if self.consumption > 0:
            lr_mixture = min(self.typeA, self.consumption-self.typeA)/(self.typeA + self.consumption-self.typeA)
        else:
            lr_mixture = 0
        return np.array([lr_capacity, lr_mixture])

    def get_obs(self, type):
        if type > 0:
            t = self.typeA
        else:
            t = 1 - self.typeA
        return [self.capacity, self.consumption, t]

    def print_resource(self):
        print(f'Beach section: {self.id}, - consumption/capacity: {self.consumption}/{self.capacity},'
              f'local reward: {self.local_reward()}\n')
