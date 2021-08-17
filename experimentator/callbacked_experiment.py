import abc
from dataclasses import dataclass
from enum import IntFlag
import itertools
import logging
import os
import re
import time
import types

import constraint as cst
import numpy as np
from mlworkflow import lazyproperty

from experimentator import BaseExperiment, SubsetType, ExperimentMode, DataCollector

# pylint: disable=logging-fstring-interpolation

class FailedTrainingError(BaseException):
    pass


@dataclass
class Callback():
    after = []
    before = []
    when = ExperimentMode.ALL
    events = ["epoch_begin", "cycle_begin", "batch_begin", "batch_end", "cycle_end", "epoch_end"]
    def fire(self, event, state):
        assert event in self.events, f"Unknown event: {event}. Existing events are {self.events}"
        cb = getattr(self, "on_{}".format(event), None)
        if cb:
            cb(**state, state=state) # pylint: disable=not-callable

class CallbackedExperiment(BaseExperiment): # pylint: disable=abstract-method
    state = {}

    @staticmethod
    def sort_callbacks(callbacks):
        # Indexing callbacks with labels in ["A", "B", "C", ...]
        callbacks = {chr(ord("A")+idx): cb for idx, cb in enumerate(callbacks) if cb is not None}
        if not callbacks:
            return []

        # Initialization of callbacks and constraint solver
        p = cst.Problem()
        for label, cb in callbacks.items():
            assert isinstance(cb, Callback), f"Only instances of 'Callback' can be used. Recieved {cb} of type {type(cb)}"
            p.addVariable(label, range(len(callbacks)))

        # Definition of constraints
        for label, other in itertools.permutations(callbacks, 2):
            for a in callbacks[label].after:
                if a in [cb.__name__ for cb in callbacks[other].__class__.__mro__]: # Check if label is in other's mro
                    p.addConstraint(lambda x, y: x > y, [label, other])
            for b in callbacks[label].before:
                if b in [cb.__name__ for cb in callbacks[other].__class__.__mro__]: # Check if label is in other's mro
                    p.addConstraint(lambda x, y: x < y, [label, other])

        # Returns callback sorted by precedence computed with the solver
        solution = p.getSolution()
        if not solution:
            raise BaseException("Impossible to solve")
        return [callbacks[label] for label, precedence in sorted(solution.items(), key=lambda kv: kv[1])]

    @lazyproperty
    def callbacks(self):
        callbacks = self.sort_callbacks(self.cfg["callbacks"])
        for cb in callbacks:
            if getattr(cb, "init", None):
                cb.init(exp=self)
        return callbacks

    def fire(self, event, mode=ExperimentMode.ALL):
        cb_messages = []
        try:
            for cb in self.callbacks:
                if cb.when & mode:
                    cb_messages.append(f"{cb}({list(self.state.keys())})")
                    cb.fire(event=event, state=self.state)
        except FailedTrainingError:
            raise
        except:
            cb_messages = "\n    ".join(cb_messages)
            logging.error(f"Error calling '{event}' after:\n    {cb_messages}")
            raise

    @lazyproperty
    def metrics(self):
        """ Describes network's outputs that should be saved in the current state
        """
        return {}

    def run_batch(self, mode, batch_id, *args, **kwargs): # pylint: disable=signature-differs
        self.state["batch"] = batch_id
        self.fire("batch_begin", mode=mode)
        result = super().run_batch(mode=mode, batch_id=batch_id, *args, **kwargs)
        self.state.update(**{k:v for k,v in result.items() if k in list(self.metrics.keys())+["loss"]})
        self.fire("batch_end", mode=mode)
        return result

    def run_cycle(self, subset, mode, *args, **kwargs):
        self.state["cycle_name"] = subset.name
        self.state["cycle_type"] = subset.type
        self.state["mode"] = mode
        self.fire("cycle_begin", mode=mode)
        super().run_cycle(subset=subset, mode=mode, *args, **kwargs)
        self.fire("cycle_end", mode=mode)
        del self.state["cycle_name"]
        del self.state["cycle_type"]
        del self.state["batch"]
        del self.state["mode"]

    def run_epoch(self, epoch, *args, **kwargs):
        self.state = {"epoch": epoch}
        self.fire("epoch_begin")
        super().run_epoch(epoch=epoch, *args, **kwargs)
        self.fire("epoch_end")

@dataclass
class SaveWeights(Callback):
    after = ["AverageLoss", "GatherCycleMetrics"]
    min_loss = None
    strategy: str = "best"
    when: IntFlag = ExperimentMode.EVAL
    def init(self, exp):
        self.exp = exp
    def on_epoch_end(self, epoch, **state):
        if     (self.strategy == "best" and (self.min_loss is None or state["training_loss"] < self.min_loss)) \
            or (self.strategy == "all"):
            self.min_loss = state.get("training_loss", None)
            filename = os.path.join(self.exp.folder, self.exp.weights_formated_filename.format(epoch=epoch))
            self.exp.save_weights(filename)

class StateLogger(Callback, metaclass=abc.ABCMeta):
    after = ["GatherCycleMetrics", "MeasureTime", "SaveLearningRate"]
    excluded_keys = ["cycle_name", "cycle_type", "mode"]
    def init(self, exp):
        self.project_name = exp.get("project_name", "unknown_project")
        self.run_name = ", ".join([f"{k}={v}" for k,v in exp.grid_sample.items()]) or "nil"
        self.config = {k:v for k,v in exp.cfg.items() if not isinstance(v, types.ModuleType)} # lists and dictionnaries don't get printed correctly

        grid_sample = dict(exp.grid_sample) # copies the original dictionary
        grid_sample.pop("fold", None)       # removes 'fold' to be able to group runs
        self.config["group"] = grid_sample
    @abc.abstractmethod
    def on_epoch_end(self, **_):
        raise NotImplementedError()

class LogStateDataCollector(StateLogger):
    @lazyproperty
    def logger(self):
        filename = os.path.join(self.config.get("folder", "."), "history.dcp")
        return DataCollector(filename)
    def on_epoch_end(self, state, **_):
        self.logger.update(**{k:v for k,v in state.items() if k not in self.excluded_keys})
        self.logger.checkpoint()

@dataclass
class AccumulateBatchMetrics(Callback):
    """ Removes metrics output by the model and accumulate them after each batch.
        When cycle ends, restore the accumulated value in state.
    """
    metrics: list = None
    def init(self, exp):
        if not self.metrics:
            self.metrics = list(exp.metrics.keys()) + ["loss"]
    def on_cycle_begin(self, **_):
        self.acc = {}
    def on_batch_end(self, state, **_): # 'state' attribute in R/W
        for name in self.metrics:
            if name in state:
                value = np.sum(state.pop(name), axis=0)
                self.acc[name] = self.acc.setdefault(name, np.zeros_like(value)) + value
    def on_cycle_end(self, state, **_): # 'state' attribute in R/W
        state.update(**self.acc) # allows augmenting metrics before gathering

@dataclass
class AverageMetrics(Callback):
    after = ["AccumulateBatchMetrics"]
    before = ["GatherCycleMetrics"]
    patterns: list
    def on_cycle_end(self, batch, state, **_):
        for name in state:
            for pattern in self.patterns:
                if re.fullmatch(pattern, name):
                    state[name] = state[name] / (batch+1)
                    break

class StopFailedTraining(Callback):
    before = ["AccumulateBatchMetrics"]
    interruption_scheduled = False
    def on_epoch_begin(self, **_):
        if self.interruption_scheduled:
            raise FailedTrainingError()
    def on_batch_end(self, loss, **_):
        if np.isnan(loss):
            self.interruption_scheduled = True

class GatherCycleMetrics(Callback):
    after = ["AccumulateBatchMetrics"]
    before = ["SaveLearningRate", "StateLogger"]
    excluded_keys = ["cycle_name", "cycle_type", "epoch", "mode", "batch"]
    def on_epoch_begin(self, **_):
        self.acc = {}
    def on_cycle_end(self, cycle_name, state, **_): # 'state' attribute in R/W
        for name in list(state):
            if name not in self.excluded_keys:
                self.acc[cycle_name + "_" + name] = state.pop(name)
    def on_epoch_end(self, state, **_): # 'state' attribute in R/W
        state.update(**self.acc)

class MeasureTime(Callback):
    after = ["GatherCycleMetrics"]
    before = ["StateLogger"]
    def on_epoch_begin(self, **_):
        self.tic_epoch = time.time()
    def on_cycle_begin(self, **_):
        self.history = []
        self.tic_cycle = time.time()
    def on_batch_begin(self, **_):
        self.tic_batch = time.time()
    def on_batch_end(self, **_):
        toc_batch = time.time()
        self.history.append(toc_batch - self.tic_batch)
    def on_cycle_end(self, state, **_):
        toc_cycle = time.time()
        state["batch_time"] = np.mean(self.history)
        state["cycle_time"] = toc_cycle - self.tic_cycle
    def on_epoch_end(self, state, **_):
        toc_epoch = time.time()
        state["epoch_time"] = toc_epoch - self.tic_epoch

class SaveLearningRate(Callback):
    after = ["GatherCycleMetrics"]
    before = ["StateLogger"]
    def init(self, exp):
        self.learning_rate_getter = exp.get_learning_rate
    def on_epoch_end(self, state, **_):
        state["learning_rate"] = self.learning_rate_getter()

@dataclass
class LearningRateDecay(Callback):
    """ Decay learning rate by a given factor spread on every batch for a given
        number of epochs.
        This callback is incompatible with a `LearningRateWarmUp` callback
        because learning rate is set from both callbacks at the beginning of
        each batch.
        start and duration are expressed in epochs here
    """
    start: (int, list, tuple)
    duration: int = 1
    factor: float = 0.1
    def __post_init__(self):
        self.start = self.start if isinstance(self.start, (list, tuple)) else ([] if self.start is None else [self.start])
    def init(self, exp):
        self.learning_rate_setter = exp.set_learning_rate
        self.learning_rate_getter = exp.get_learning_rate
        self.learning_rate = self.learning_rate_getter()
        self.batch_count = sum([len(subset.keys)//exp.batch_size for _, subset in exp.subsets.items() if subset.type & SubsetType.TRAIN])
        # adjust start and duration per step
        self.start = [epoch_start * self.batch_count for epoch_start in self.start]
        self.duration = self.duration * self.batch_count
        assert all([not isinstance(cb, LearningRateWarmUp) for cb in exp.cfg['callbacks']]), f"{self.__class__} is incompatible with LearningRateWarmUp"
    def compute_factor(self, step):
        step_factor = 1.0
        for start in self.start:
            stop = start + self.duration
            if step >= stop: # after decay period
                step_factor *= self.factor
            elif start < step < stop: # during decay period
                step_factor *= self.factor ** ((step - start)/self.duration)
        return step_factor
    def on_batch_begin(self, cycle_type, epoch, batch, **_):
        if cycle_type == SubsetType.TRAIN and self.start:
            self.learning_rate_setter(self.learning_rate * self.compute_factor(step=epoch*self.batch_count + batch))

@dataclass
class LearningRateWarmUp(Callback):
    """ Decay learning rate by a given factor spread on every batch for a given
        number of epochs.
        This callback is incompatible with a `LearningRateDecay` callback
        because learning rate is set from both callbacks at the beginning of
        each batch.
        start and duration are expressed in epochs here
    """
    # TODO: check tf.keras.optimizers.schedules.LearningRateSchedule
    start: int = 0
    duration: int = 2
    factor: float = 0.001
    def init(self, exp):
        self.learning_rate_setter = exp.set_learning_rate
        self.learning_rate_getter = exp.get_learning_rate
        self.learning_rate = self.learning_rate_getter()
        self.batch_count = sum([len(subset.keys)//exp.batch_size for _, subset in exp.subsets.items() if subset.type & SubsetType.TRAIN])
        # adjust start and duration per step
        self.start *= self.batch_count
        self.duration *= self.batch_count
        assert all([not isinstance(cb, LearningRateDecay) for cb in exp.cfg['callbacks']]), f"{self.__class__} is incompatible with LearningRateDecay"
    def compute_factor(self, step):
        step_factor = 1.0
        if step <= self.start:
            step_factor *= self.factor
        elif self.start < step < self.start+self.duration:
            step_factor *= self.factor ** (1.0 - (step - self.start)/self.duration)
        return step_factor
    def on_batch_begin(self, cycle_type, epoch, batch, **_):
        if cycle_type == SubsetType.TRAIN:
            self.learning_rate_setter(self.learning_rate * self.compute_factor(step=epoch*self.batch_count + batch))

@dataclass
class PrintLoss(Callback):
    """ Prints current batch loss every 'period' seconds
    """
    before = ["AccumulateBatchMetrics"]
    period: int = 10
    def __post_init__(self):
        self.tic = time.time()
    def on_batch_end(self, epoch, batch, loss, **_):
        tac = time.time()
        if tac - self.tic > self.period:
            print(f"Epoch:{epoch} Batch:{batch} - batch loss={loss}")
            self.tic = tac
