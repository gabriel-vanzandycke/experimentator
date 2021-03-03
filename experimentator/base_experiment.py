import logging
from tqdm.auto import tqdm

from mlworkflow import SideRunner, lazyproperty, TransformedDataset, PickledDataset
from .utils import find, RobustBatchesDataset, insert_suffix

class BaseExperiment():
    batch_count = 0
    def __init__(self, config):
        self.cfg = config

    def __del__(self):
        pass

    def update(self, **kwargs):
        self.cfg.update(**kwargs)

    def get(self, key, default):
        return self.cfg.get(key, default)

    @lazyproperty
    def logger(self):
        return logging.getLogger()

    @property
    def epochs(self):
        return 0 # can be overwritten in a LoggedExperiment to continue a loaded traning

    @lazyproperty
    def grid_sample(self):
        return self.cfg["grid_sample"]

    @lazyproperty
    def dataset(self):
        dataset_name = self.cfg["dataset_name"]
        dataset = PickledDataset(find(dataset_name, verbose=False))
        dataset = TransformedDataset(dataset, self.get("transforms", []))
        dataset = RobustBatchesDataset(dataset)
        return dataset

    @lazyproperty
    def subsets(self):
        keys_splitter = self.cfg["keys_splitter"]
        return keys_splitter(self.dataset.keys, fold=self.get("fold", default=0))

    def progress(self, generator, **kwargs):
        return tqdm(generator, **kwargs, disable=self.get("hide_progress", False), leave=False)

    @property
    def batch_size(self):
        return self.cfg["batch_size"]

    def batch_generator(self, keys, batch_size=None):
        batch_size = batch_size or self.batch_size
        self.batch_count += len(keys)//batch_size
        for keys, batch in self.dataset.batches(keys, batch_size, drop_incomplete=True):
            yield keys, {"batch_{}".format(k): v for k,v in batch.items()}

    def batch_infer(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")
    def batch_train(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")
    def batch_eval(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")

    def run_batch(self, subset, data):
        if subset.mode == "TRAIN":
            return self.batch_train(data)
        elif subset.mode == "EVAL":
            return self.batch_eval(data)
        elif subset.mode == "INFER":
            return self.batch_infer(data)
        else:
            raise NotImplementedError("Mode undefined: '{}' for subset '{}'".format(subset.mode, subset.name))

    def run_cycle(self, subset, progress):
        progress.set_description(subset.name)
        for keys, data in self.batch_generator(subset.shuffeled_keys): # pylint: disable=unused-variable
            _ = self.run_batch(subset=subset, data=data)
            progress.update(1)

    def run_epoch(self, epoch):
        progress = self.progress(None, total=self.batch_count, unit="batches")
        self.batch_count = 0  # required
        for subset_name, subset in self.subsets.items():
            assert subset.keys, "Empty subset is not allowed: {}".format(subset_name)
            if subset.do_run_epoch(epoch):
                self.run_cycle(subset, progress)
        progress.close()

    def train(self, epochs):
        range_epochs = range(self.epochs+1, epochs+1)
        for epoch in self.progress(range_epochs, desc="epochs"):
            self.run_epoch(epoch)

    def predict(self, data):
        return self.batch_infer(data)


class AsyncExperiment(BaseExperiment): # pylint: disable=abstract-method
    @lazyproperty
    def side_runner(self):
        return SideRunner()

    def batch_generator(self, *args, **kwargs): # pylint: disable=signature-differs
        batch_generator = super().batch_generator(*args, **kwargs)
        return self.side_runner.yield_async(batch_generator)


class DummyExperiment(BaseExperiment): # pylint: disable=abstract-method
    def batch_generator(self, *args, **kwargs): # pylint: disable=signature-differs
        self.cfg["dummy"] = True
        gen = super().batch_generator(*args, **kwargs)
        # yield only two batches per cycle
        for _ in range(3):
            yield next(gen)
