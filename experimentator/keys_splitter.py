import random
from utils import linestyles, ExperimentMode, Subset


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
            "training": Subset("training", type=ExperimentMode.TRAIN, keys=training_keys),
            "validation": Subset("validation", type=ExperimentMode.EVAL, keys=validation_keys),
            "testing": Subset("testing", type=ExperimentMode.EVAL, keys=testing_keys),
        }
