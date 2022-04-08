import dataclasses
from enum import IntFlag
import random
import numpy as np

from mlworkflow import Dataset, FilteredDataset
from mlworkflow.datasets import batchify
from aleatorpy import pseudo_random, method # pylint: disable=unused-import

def collate_fn(items):
    return {f"batch_{k}": v for k,v in batchify(items).items()}

# This object is defined both in here and in dataset-utilities repository
# Any change here should be reported in dataset-utilities as well
class SubsetType(IntFlag):
    TRAIN = 1
    EVAL  = 2

# This object is defined both in here and in dataset-utilities repository
# Any change here should be reported in dataset-utilities as well
class Subset:
    def __init__(self, name: str, subset_type: SubsetType, dataset: Dataset, keys=None, repetitions=1, desc=None):
        keys = keys if keys is not None else dataset.keys.all()
        assert isinstance(keys, (tuple, list)), f"Received instance of {type(keys)} for subset {name}"
        self.name = name
        self.type = subset_type
        self.dataset = FilteredDataset(dataset, predicate=lambda k,v: v is not None)
        self._keys = keys
        self.keys = keys
        self.repetitions = repetitions
        self.desc = desc
        self.is_training = self.type == SubsetType.TRAIN
        loop = None if self.is_training else repetitions
        self.shuffled_keys = pseudo_random(evolutive=self.is_training)(self.shuffled_keys)
        self.dataset.query_item = pseudo_random(loop=loop, input_dependent=True)(self.dataset.query_item)

    def shuffled_keys(self): # pylint: disable=method-hidden
        keys = self.keys * self.repetitions
        return random.sample(keys, len(keys)) if self.is_training else keys

    def __len__(self):
        return len(self.keys)*self.repetitions

from itertools import chain
class CombinedSubsets:
    def __init__(self, *subsets):
        self.subsets = subsets
    def shuffled_keys(self):
        return list(chain(*zip(*[s.shuffled_keys() for s in self.subsets])))
    def __len__(self):
        return sum([len(s) for s in self.subsets])

@dataclasses.dataclass
class BasicDatasetSplitter:
    validation_pc: int = 15
    testing_pc: int = 15
    def __post_init__(self):
        assert self.validation_pc + self.testing_pc < 100

    def __call__(self, dataset, fold=0):
        keys = list(dataset.keys.all())
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

        return [
            Subset("training", subset_type=SubsetType.TRAIN, keys=training_keys, dataset=dataset),
            Subset("validation", subset_type=SubsetType.EVAL, keys=validation_keys, dataset=dataset),
            Subset("testing", subset_type=SubsetType.EVAL, keys=testing_keys, dataset=dataset),
        ]
