import dataclasses
from enum import IntFlag
import random
import numpy as np

from mlworkflow import Dataset
from mlworkflow.datasets import batchify
from aleatorpy import pseudo_random, method # pylint: disable=unused-import


# This object is defined both in here and in dataset-utilities repository
# Any change here should be reported in dataset-utilities as well
class RobustDataset(Dataset):
    """Handles a parent dataset whose `query_item` can return `None`.
    """
    def __init__(self, parent):
        self.parent = parent
    def yield_keys(self):
        return self.parent.keys
    def query_item(self, key, retry=0):
        item = self.parent.query_item(key)
        while item is None and retry:
            item = self.parent.query_item(key)
            retry -= 1
        return item
    def chunkify(self, keys, chunk_size, drop_incomplete=True, retry=0):
        d = []
        for k, v in ((k, v) for k,v in ((k, self.query_item(k, retry=retry)) for k in keys) if v is not None):
            d.append((k, v))
            if len(d) == chunk_size:  # yield complete sublist and create a new list
                yield d
                d = []
        assert drop_incomplete, "not implemented"
    def batches(self, batch_size, keys=None, wrapper=np.array, drop_incomplete=False, retry=0):
        keys = keys or self.keys
        for chunk in self.chunkify(keys, chunk_size=batch_size, drop_incomplete=drop_incomplete, retry=retry):
            keys, batch = list(zip(*chunk)) # transforms list of (k,v) into list of (k) and list of (v)
            yield keys, batchify(batch, wrapper=wrapper)



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
        self.dataset = RobustDataset(dataset)
        self._keys = keys
        self.keys = keys
        self.repetitions = repetitions
        self.desc = desc
        evolutive = self.type == SubsetType.TRAIN
        loop = None if evolutive else repetitions
        self.shuffled_keys = pseudo_random(evolutive=evolutive)(self.shuffled_keys)
        self.dataset.query_item = pseudo_random(loop=loop, input_dependent=True)(self.dataset.query_item)

    def shuffled_keys(self): # pylint: disable=method-hidden
        keys = self.keys * self.repetitions
        return random.sample(keys, len(keys))

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
