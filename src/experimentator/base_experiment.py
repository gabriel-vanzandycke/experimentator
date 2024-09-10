import abc
from functools import cached_property
import logging

import numpy as np
from tqdm.auto import tqdm
from mlworkflow import SideRunner

from experimentator.dataset import Subset, SubsetType
from experimentator.utils import ExperimentMode

# pylint: disable=abstract-method

class BaseExperiment(metaclass=abc.ABCMeta):
    epoch = 0
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

    @cached_property
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
        return self.get('output_folder', 'output_folder')

    @property
    def epochs(self):
        return 0 # can be overwritten in a LoggedExperiment to continue a loaded traning

    @cached_property
    def grid_sample(self):
        return self.cfg.get("grid_sample", {})

    @cached_property
    def subsets(self):
        subsets = []
        for subset in self.cfg["subsets"]:
            if subset.keys:
                subsets.append(subset)
                print(subset)
            else:
                print("Skipping empty subset: {}".format(subset.name))
        return subsets

    def progress(self, generator, leave=False, **kwargs):
        return tqdm(generator, **kwargs, disable=self.get("hide_progress", False), leave=leave)

    @property
    def batch_size(self):
        return self.cfg["batch_size"]

    # yields pairs of (keys, data)
    def batch_generator(self, subset: Subset, batch_size=None, **kwargs):
        batch_size = batch_size or self.batch_size
        drop_last = subset.type == SubsetType.TRAIN
        yield from subset.batches(batch_size=batch_size, drop_last=drop_last, **kwargs)

    @abc.abstractmethod
    def batch_infer(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")
    @abc.abstractmethod
    def batch_train(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")
    @abc.abstractmethod
    def batch_eval(self, *args, **kwargs):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")

    def run_batch(self, subset: Subset, batch_id: int, batch_generator, mode=ExperimentMode.ALL): # pylint: disable=unused-argument
        keys, data = next(batch_generator) # pylint: disable=unused-variable
        data['epoch'] = np.array([self.epoch]*len(keys))
        if subset.type == SubsetType.TRAIN:
            return keys, data, self.batch_train(data, mode)
        elif subset.type == SubsetType.EVAL:
            return keys, data, self.batch_eval(data)
        raise ValueError("Unknown subset type: {}".format(subset.type))

    def run_cycle(self, subset: Subset, mode: ExperimentMode, epoch_progress):
        epoch_progress.set_description(subset.name)
        batch_generator = iter(self.batch_generator(subset=subset, drop_incomplete=True))
        batch_id = 0
        while True:
            try:
                _ = self.run_batch(subset=subset, batch_id=batch_id, batch_generator=batch_generator, mode=mode)
                epoch_progress.update(1)
                batch_id += 1
            except StopIteration:
                break
            except RuntimeError:
                break

    def run_epoch(self, mode: ExperimentMode): # pylint: disable=unused-argument
        cond = lambda subset: mode == ExperimentMode.EVAL or subset.type == SubsetType.TRAIN
        subsets = [subset for subset in self.subsets if cond(subset)]
        assert subsets, "No single subset for this epoch"
        epoch_progress = self.progress(None, total=sum([len(subset)//self.batch_size for subset in subsets]), unit="batches")
        for subset in subsets:
            self.run_cycle(subset=subset, mode=mode, epoch_progress=epoch_progress)
        epoch_progress.close()

    def train(self, epochs):
        range_epochs = range(self.epochs, epochs)
        for self.epoch in self.progress(range_epochs, leave=True, desc="epochs"):
            eval_epochs = self.cfg.get("eval_epochs", [self.epoch])
            mode = ExperimentMode.EVAL if self.epoch in eval_epochs else ExperimentMode.TRAIN
            self.run_epoch(mode=mode)

    def predict(self, data):
        return self.batch_infer(data)

class AsyncExperiment(BaseExperiment):
    @cached_property
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
