import dataclasses
from enum import IntFlag
import errno
import logging
import os
import random
import shutil
import threading

import numpy as np

from mlworkflow.datasets import batchify, Dataset, AugmentedDataset, PickledDataset
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
        self.dataset = dataset#FilteredDataset(dataset, predicate=lambda k,v: v is not None)
        self._keys = keys
        self.keys = keys
        self.repetitions = repetitions
        self.desc = desc
        self.is_training = self.type == SubsetType.TRAIN
        loop = None if self.is_training else repetitions
        self.shuffled_keys = pseudo_random(evolutive=self.is_training)(self.shuffled_keys)
        self.query_item = pseudo_random(loop=loop, input_dependent=True)(self.query_item)

    def shuffled_keys(self): # pylint: disable=method-hidden
        keys = self.keys * self.repetitions
        return random.sample(keys, len(keys)) if self.is_training else keys

    def __len__(self):
        return len(self.keys)*self.repetitions

    def __str__(self):
        return f"{self.__class__.__name__}<{self.name}>({len(self)})"

    def query_item(self, key):
        return self.dataset.query_item(key)

    def chunkify(self, keys, chunk_size):
        d = []
        for k in keys:
            try:
                v = self.query_item(k)
            except KeyError:
                continue
            if v is None:
                continue
            d.append((k, v))
            if len(d) == chunk_size:  # yield complete sublist and create a new list
                yield d
                d = []

    def batches(self, batch_size, keys=None, *args, **kwargs):
        keys = keys or self.shuffled_keys()
        for chunk in self.chunkify(keys, chunk_size=batch_size):
            keys, batch = list(zip(*chunk)) # transforms list of (k,v) into list of (k) and list of (v)
            yield keys, collate_fn(batch)


class FastFilteredDataset(Dataset):
    def __init__(self, parent, predicate):
        self.parent = parent
        self.predicate = predicate
        self.cached_keys = list(self.parent.keys.all())

    def yield_keys(self):
        yield from self.cached_keys

    def __len__(self):
        return len(self.cached_keys)

    def query_item(self, key):
        try:
            item = self.parent.query_item(key)
            if self.predicate(key, item):
                return item
        except KeyError:
            pass
        self.cached_keys.remove(key)
        return None


class CombinedSubset(Subset):
    def __init__(self, name, *subsets):
        self.subsets = subsets
        self.name = name
        assert len(set(subset.type for subset in subsets)) == 1, "Combined Subsets must have the same type"
        self.type = subsets[0].type

    def __len__(self):
        return min(len(subset) for subset in self.subsets)*len(self.subsets)

    def batches(self, batch_size, **kwargs):
        assert batch_size % len(self.subsets) == 0, f"Batch size must be a multiple of the number of subsets ({len(self.subsets)})"
        batch_size = batch_size // len(self.subsets)
        iterators = [subset.batches(batch_size, **kwargs) for subset in self.subsets]
        while True:
            try:
                key_chunks, chunks = zip(*[next(it) for it in iterators])
            except StopIteration:
                break
            keys = [key for key_chunk in key_chunks for key in key_chunk]
            batch = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
            yield keys, batch


class BalancedSubest(Subset):
    """
    """
    def __init__(self, balancing_attr, *args, **kwargs):
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


# class UnSafeCachedPickledDataset(PickledDataset):
#     def __init__(self, filename, local_scratch=""):
#         super().__init__(filename)
#         local_scratch = local_scratch or os.environ.get('LOCALSCRATCH', "")
#         self.filename = None
#         if local_scratch:
#             def f():
#                 print("Copying dataset to local scratch...")
#                 self.filename = shutil.copy(filename, local_scratch)
#                 print("Done.")
#             self.sr = SideRunner()
#             self.sr.do(f)
#         else:
#             self.query_item = super().query_item

#     def query_item(self, key):
#         if self.filename:
#             super().__init__(self.filename)
#             self.query_item = super().query_item
#         return super().query_item(key)

class CachedPickledDataset(PickledDataset):
    def __init__(self, filename, local_scratch=""):
        super().__init__(filename)
        local_scratch = local_scratch or os.environ.get('LOCALSCRATCH', "")
        if not local_scratch or local_scratch in filename:
            self.query_item = super().query_item
            return

        self.filename = os.path.join(local_scratch, os.path.basename(filename))
        lock = f"{self.filename}.lock"
        self.available = lambda: not os.path.exists(lock)

        try:
            with open(lock, "x") as _:
                pass # First process to reach this point copies the dataset
        except FileExistsError:
            return

        if os.path.isfile(self.filename):
            os.remove(lock)
            self.reload()
            return

        def function():
            logging.info(f"Copying dataset to local scratch: {filename} -> " \
                f"{self.filename}.")
            try:
                shutil.copy(filename, self.filename)
            except:
                try:
                    os.remove(self.filename)
                except:
                    pass
                logging.info("Failed copying dataset.")
                self.query_item = super().query_item
            os.remove(lock)
        self.thread = threading.Thread(target=function, daemon=True)
        self.thread.start()

    def reload(self):
        logging.info(f"Reloading dataset from {self.filename}")
        super().__init__(self.filename)
        self.query_item = super().query_item

    def query_item(self, key):
        if self.available():
            self.reload()
        return super().query_item(key)



def find(path, dirs=None, verbose=True, fail=True):
    if os.path.isabs(path):
        if not os.path.isfile(path) and not os.path.isdir(path):
            if not fail:
                not verbose or print(f"{path} not found")
                return None
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        return path

    dirs = dirs or [os.getcwd(), *os.getenv("DATA_PATH", "").split(":")]
    for dirname in dirs:
        if dirname is None:
            continue
        tmp_path = os.path.join(dirname, path)
        if os.path.isfile(tmp_path) or os.path.isdir(tmp_path):
            not verbose or print("{} found in {}".format(path, tmp_path))
            return tmp_path

    if not fail:
        not verbose or print(f"{path} not found (searched in {dirs})")
        return None
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                "{} (searched in {})".format(path, dirs))
