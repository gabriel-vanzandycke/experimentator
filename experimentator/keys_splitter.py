import random

from utils import linestyles

class Subset():
    random_count = 0
    def __init__(self, name, type, keys, repetitions=1, legend=None):
        self.name = name
        self.type = type
        self.keys = keys
        self.linestyle = linestyles.get(name, "dotted")
        self.repetitions = 1 if type == "TRAIN" else repetitions
        self.legend = legend or self.name

    @property
    def shuffeled_keys(self):
        random_state = random.getstate()
        random.seed(self.random_count)
        shuffeled_keys = list(self.keys)*self.repetitions
        random.shuffle(shuffeled_keys)
        random.setstate(random_state)
        self.random_count += 1
        return shuffeled_keys


class KeysSplitter(): # pylint: disable=too-few-public-methods
    def __str__(self):
        return self.__class__.__name__
