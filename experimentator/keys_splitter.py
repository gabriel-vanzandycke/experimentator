import random

from utils import linestyles

class Subset():
    random_count = 0
    def __init__(self, name, mode, keys, frequency=1, repetitions=1, legend=None):
        self.name = name
        self.mode = mode
        self.keys = keys
        self.linestyle = linestyles.get(name, "dotted")
        self.frequency = 1 if mode == "TRAIN" else frequency
        self.repetitions = 1 if mode == "TRAIN" else repetitions
        self.legend = legend or self.name

    def do_run_epoch(self, epoch):
        if (epoch % self.frequency) != 0:
            return False
        return True

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
    pass