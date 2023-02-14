import dataclasses
from enum import IntFlag
import errno
import os
import random

from mlworkflow.datasets import batchify, Dataset, FilteredDataset, AugmentedDataset
from aleatorpy import pseudo_random, method # pylint: disable=unused-import




def collate_fn(items):
    return {f"batch_{k}": v for k,v in batchify(items).items()}


# This object is defined both in here and in experimentator repository
# Any change here should be reported in experimentator as well
class SubsetType(IntFlag):
    TRAIN = 1
    EVAL  = 2

# This object is defined both in here and in experimentator repository
# Any change here should be reported in experimentator as well
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

    def __str__(self):
        return f"Subset<{self.name}>({len(self)})"

    def batches(self, batch_size, keys=None, *args, **kwargs):
        keys = keys or self.shuffled_keys()
        yield from self.dataset.batches(batch_size=batch_size, keys=keys, collate_fn=collate_fn, *args, **kwargs)


class CombinedSubset(Subset):
    def __init__(self, name, subsets):
        self.subsets = subsets
         # must
        self.name = name
        self.type = SubsetType.TRAIN if any(s.type == SubsetType.TRAIN for s in subsets) else SubsetType.EVAL

        self.is_training = self.type == SubsetType.TRAIN
        self.datasets = [subset.dataset for subset in subsets]
        self.keys = [subset.keys for subset in subsets]
        self.repetitions = 1

    def __len__(self):
        raise NotImplementedError

    def shuffled_keys(self): # pylint: disable=method-hidden
        raise NotImplementedError

    def __str__(self):
        return f"CombinedSubset<{self.name}>({len(self)})"



class BalancedSubest(Subset):
    """
    """
    def __init(self, balancing_attr, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.balancing_attr = balancing_attr
    def shuffled_keys(self):
        # logic
        return None

class MergedDataset(Dataset):
    def __init__(self, *ds):
        self.ds = ds
        self.cache = {}
    def yield_keys(self):
        for ds in self.ds:
            for key in ds.yield_keys():
                self.cache[key] = ds
                yield key
    def query_item(self, key):
        return self.cache[key].query_item(key)



class TolerentDataset(AugmentedDataset):
    def __init__(self, parent, retry=0):
        super().__init__(parent)
        self.retry = retry
    def augment(self, root_key, root_item):
        retry = self.retry
        while root_item is None and retry:
            root_item = self.parent.query_item(root_key)
            retry -= 1
        return root_item



class DatasetSamplerDataset(Dataset):
    def __init__(self, dataset, count):
        self.parent = dataset
        self.keys = random.sample(list(dataset.keys.all()), count)
    def yield_keys(self):
        for key in self.keys:
            yield key
    def query_item(self, key):
        return self.parent.query_item(key)


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


def split_equally(d, K):
    """ splits equally the keys of d given their values
        arguments:
            d (dict) - A dict {"label1": 30, "label2": 45, "label3": 22, ... "label<N>": 14}
            K (int)  - The number of split to make
        returns:
            A list of 'K' lists splitting equally the values of 'd':
            e.g. [[label1, label12, label19], [label2, label15], [label3, label10, label11], ...]
            where
            ```
               d["label1"]+d["label12"]+d["label19"]  ~=  d["label2"]+d["label15"]  ~=  d["label3"]+d["label10"]+d["label11]
            ```
    """
    s = sorted(d.items(), key=lambda kv: kv[1])
    f = [{"count": 0, "list": []} for _ in range(K)]
    while s:
        arena_label, count = s.pop(-1)
        index, _ = min(enumerate(f), key=(lambda x: x[1]["count"]))
        f[index]["count"] += count
        f[index]["list"].append(arena_label)
    return [x["list"] for x in f]



def find(path, dirs=None, verbose=True):
    if os.path.isabs(path):
        if not os.path.isfile(path) and not os.path.isdir(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return path

    dirs = dirs or [os.getcwd(), *os.getenv("DATA_PATH", "").split(":")]
    for dirname in dirs:
        if dirname is None:
            continue
        tmp_path = os.path.join(dirname, path)
        if os.path.isfile(tmp_path) or os.path.isdir(tmp_path):
            if verbose:
                print("{} found in {}".format(path, tmp_path))
            return tmp_path

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                "{} (searched in {})".format(path, dirs))
