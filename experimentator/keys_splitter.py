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


class BasicKeysSplitter(KeysSplitter):
    def __init__(self, validation_pc=15, testing_pc=15):
        assert validation_pc + testing_pc < 100
        self.validation_pc = validation_pc
        self.testing_pc = testing_pc

    def __call__(self, keys, fold=0):
        keys = list(keys.all())
        l = len(keys)

        # Backup random seed
        random_state = random.getstate()
        random.seed(fold)

        random.shuffle(keys)

        # Restore random seed
        random.setstate(random_state)

        u1 = self.validation_pc
        u2 = self.validation_pc + self.testing_pc

        validation_keys = keys[00*l//100:u1*l//100]
        testing_keys    = keys[u1*l//100:u2*l//100]
        training_keys   = keys[u2*l//100:]

        return {
            "training": Subset("training", "TRAIN", training_keys),
            "validation": Subset("validation", "VALID", validation_keys, repetitions=5),
            "testing": Subset("testing", "TEST", testing_keys, repetitions=5),
        }
