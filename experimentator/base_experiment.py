import abc
import logging

from tqdm.auto import tqdm
from mlworkflow import SideRunner, lazyproperty

from .dataset import Subset, SubsetType
from .utils import ExperimentMode

# pylint: disable=abstract-method

class BaseExperiment(metaclass=abc.ABCMeta):
    batch_count = 0
    def __init__(self, config):
        self.cfg = config

    def __del__(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def update(self, **kwargs):
        self.cfg.update(**kwargs)

    def get(self, key, default):
        return self.cfg.get(key, default)

    @lazyproperty
    def logger(self):
        return logging.getLogger()

    @property
    def project_name(self):
        return self.get('project_name', 'unknown_project')

    @property
    def experiment_id(self):
        return self.get('experiment_id', 0)

    @property
    def folder(self):
        return self.get('folder', 'default_folder')

    @property
    def epochs(self):
        return 0 # can be overwritten in a LoggedExperiment to continue a loaded traning

    @lazyproperty
    def grid_sample(self):
        return self.cfg["grid_sample"]

    @lazyproperty
    def subsets(self):
        return self.cfg["subsets"]

    def progress(self, generator, **kwargs):
        return tqdm(generator, **kwargs, disable=self.get("hide_progress", False), leave=False)

    @property
    def batch_size(self):
        return self.cfg["batch_size"]

    def batch_generator(self, subset: Subset, batch_size=None):
        batch_size = batch_size or self.batch_size
        self.batch_count += len(subset)//batch_size
        for keys, batch in subset.dataset.batches(subset.shuffled_keys(), batch_size, drop_incomplete=True):
            yield keys, {"batch_{}".format(k): v for k,v in batch.items()}

    @abc.abstractmethod
    def batch_infer(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")
    @abc.abstractmethod
    def batch_train(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")
    @abc.abstractmethod
    def batch_eval(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")

    def run_batch(self, subset, batch_id, batch_generator, mode=ExperimentMode.ALL): # pylint: disable=unused-argument
        keys, data = next(batch_generator) # pylint: disable=unused-variable
        if subset.type == SubsetType.TRAIN:
            return self.batch_train(data, mode)
        elif subset.type == SubsetType.EVAL:
            return self.batch_eval(data)
        else:
            raise ValueError("Subset type should either be {} or {}".format(SubsetType.TRAIN, SubsetType.EVAL))

    def run_cycle(self, subset: Subset, mode: ExperimentMode, epoch_progress):
        epoch_progress.set_description(subset.name)
        batch_generator = iter(self.batch_generator(subset=subset))
        batch_id = 0
        while True:
            try:
                _ = self.run_batch(subset=subset, batch_id=batch_id, batch_generator=batch_generator, mode=mode)
                epoch_progress.update(1)
                batch_id += 1
            except StopIteration:
                break

    def run_epoch(self, epoch):
        epoch_progress = self.progress(None, total=self.batch_count, unit="batches")
        self.batch_count = 0  # required
        for subset_name, subset in self.subsets.items():
            assert subset.keys, "Empty subset is not allowed because it would require to adapt all callacks: {}".format(subset_name)
            eval_epochs = self.cfg.get("eval_epochs", None)
            if eval_epochs is None or epoch in eval_epochs:
                mode = ExperimentMode.EVAL
            elif subset.type == SubsetType.TRAIN:
                mode = ExperimentMode.TRAIN
            else:
                continue # skip this cycle for this epoch
            self.run_cycle(subset=subset, mode=mode, epoch_progress=epoch_progress)
        epoch_progress.close()

    def train(self, epochs):
        range_epochs = range(self.epochs, epochs)
        for epoch in self.progress(range_epochs, desc="epochs"):
            self.run_epoch(epoch=epoch)

    def predict(self, data):
        return self.batch_infer(data)

class AsyncExperiment(BaseExperiment):
    @lazyproperty
    def side_runner(self):
        return SideRunner()

    def batch_generator(self, *args, **kwargs): # pylint: disable=signature-differs
        batch_generator = super().batch_generator(*args, **kwargs)
        return self.side_runner.yield_async(batch_generator)

class DummyExperiment(BaseExperiment):
    def batch_generator(self, *args, **kwargs): # pylint: disable=signature-differs
        self.cfg["dummy"] = True
        gen = super().batch_generator(*args, **kwargs)
        # yield only two batches per cycle
        for _ in range(3):
            yield next(gen)
