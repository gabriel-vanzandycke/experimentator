import abc
from dataclasses import dataclass
from enum import IntFlag
from functools import cached_property
import itertools
import logging
import os
import re
import time
import types
from typing import Tuple

import pickle
import constraint as cst
import numpy as np

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

    @cached_property
    def callbacks(self):
        callbacks = self.sort_callbacks(self.cfg["callbacks"])
        try:
            for callback in callbacks:
                if getattr(callback, "init", None):
                    callback.init(exp=self)
        except:
            logging.error(f"Error initializing '{callback}'")
            raise
        #     message = []
        #     list(map(lambda cb: message.append(("  * " if cb == callback else "   ")+ str(cb)), callbacks))
        #     print("\n".join(message))
        #     raise
        return callbacks

    def fire(self, event, state, mode=ExperimentMode.ALL):
        cb_messages = []
        try:
            for cb in self.callbacks:
                if cb.when & mode:
                    cb_messages.append(f"{cb}({list(state.keys())})")
                    cb.fire(event=event, state=state)
        except FailedTrainingError:
            raise
        except:
            cb_messages = "\n    ".join(cb_messages)
            logging.error(f"Error calling '{event}' after:\n    {cb_messages}")
            raise

    def run_batch(self, mode, batch_id, *args, **kwargs): # pylint: disable=signature-differs
        state = dict(**self.state)
        state["batch"] = batch_id
        self.fire("batch_begin", state=state, mode=mode)
        keys, data, result = super().run_batch(mode=mode, batch_id=batch_id, *args, **kwargs)
        state.update(keys=keys, **{**data, **{k: v.numpy() for k,v in result.items()}})
        #self.state.update(**{k:v for k,v in result.items() if k in list(self.metrics.keys())+["loss"]})
        self.fire("batch_end", state=state, mode=mode)
        return result

    def run_cycle(self, subset, mode, *args, **kwargs):
        self.state["cycle_name"] = subset.name
        self.state["cycle_type"] = subset.type
        self.state["mode"] = mode
        state = dict(**self.state)
        self.fire("cycle_begin", state=state, mode=mode)
        super().run_cycle(subset=subset, mode=mode, *args, **kwargs)
        self.fire("cycle_end", state=state, mode=mode)
        del self.state["cycle_name"]
        del self.state["cycle_type"]
        del self.state["mode"]

    def run_epoch(self, epoch, mode, *args, **kwargs):
        self.state = {"epoch": epoch}
        # state = dict(**self.state)
        self.fire("epoch_begin", state=self.state, mode=mode)
        super().run_epoch(epoch=epoch, mode=mode, *args, **kwargs)
        self.fire("epoch_end", state=self.state, mode=mode)
        # del self.state["epoch"]

@dataclass
class SaveWeights(Callback):
    after = ["GatherCycleMetrics"]
    metric: str = "validation_loss"
    strategy: str = "all"
    when: IntFlag = ExperimentMode.EVAL
    def init(self, exp):
        self.exp = exp
        self.best = None
    def do_save_weights(self, state):
        if self.strategy == "all":
            return True
        current = state[self.metric]
        if self.strategy == "min":
            operator = min
        elif self.strategy == "max":
            operator = max
        else:
            raise ValueError("Unknown strategy. Expected 'all', 'min' or 'max'")
        if self.best is None or operator(self.best, current) == current:
            self.best = current
            return True
        return False
    def on_epoch_end(self, epoch, **state):
        if self.do_save_weights(state):
            filename = os.path.join(self.exp.folder, self.exp.weights_formated_filename.format(epoch=epoch))
            self.exp.train_model.save_weights(filename)

@dataclass
class LoadWeights(Callback):
    folder: str
    def init(self, exp):
        self.exp = exp
    def on_epoch_begin(self, epoch, **_):
        filename = os.path.join(self.folder, self.exp.weights_formated_filename.format(epoch=epoch))
        self.exp.eval_model.load_weights(filename)

@dataclass
class StateLogger(Callback, metaclass=abc.ABCMeta):
    after = ["GatherCycleMetrics", "MeasureTime", "SaveLearningRate"]
    def init(self, exp):
        self.project_name = exp.get("project_name", "unknown_project")
        self.run_name = ", ".join([f"{k}={v}" for k,v in exp.grid_sample.items()]) or exp.get("run_name", "nil")
        self.config = {k:v for k,v in exp.cfg.items() if not isinstance(v, types.ModuleType)} # lists and dictionnaries don't get printed correctly

        grid_sample = dict(exp.grid_sample) # copies the original dictionary
        grid_sample.pop("fold", None)       # removes 'fold' to be able to group runs
        grid_sample.pop("init_index", None) # removes 'fold' to be able to group runs
        self.config["group"] = grid_sample
    @abc.abstractmethod
    def on_epoch_end(self, **_):
        raise NotImplementedError()

class LogStateDataCollector(StateLogger):
    @cached_property
    def logger(self):
        filename = os.path.join(self.config.get("output_folder", "."), "history.dcp")
        return DataCollector(filename)
    def on_epoch_end(self, state, **_):
        self.logger.update(**state)
        self.logger.checkpoint()

@dataclass
class AverageMetrics(Callback):
    before = ["GatherCycleMetrics"]
    patterns: list
    def on_cycle_begin(self, **_):
        self.acc = {}
    def matching_patterns(self, state: dict):
        for name in state:
            for pattern in self.patterns:
                if re.fullmatch(pattern, name):
                    yield name
    def on_batch_end(self, **state):
        for name in self.matching_patterns(state):
            self.acc.setdefault(name, []).append(state[name])

    def on_cycle_end(self, state: dict, **_): # 'state' argument in R/W
        for name, value in self.acc.items():
            state[name] = np.mean(np.stack(value, axis=0), axis=0)

@dataclass
class StopFailedTraining(Callback):
    before = ["StateLogger"]
    interruption_scheduled = False
    consecutive_nans: int = 1
    def __post_init__(self):
        self.default_consecutive_nans = self.consecutive_nans
    def on_epoch_begin(self, **_):
        if self.interruption_scheduled:
            raise FailedTrainingError()
    def on_batch_end(self, batch, cycle_type, epoch, loss, **state):
        if np.isnan(loss) and cycle_type == SubsetType.TRAIN:
            self.consecutive_nans -= 1
            print(f"NaNs detected at epoch{epoch}, batch{batch}. {state}. consecutive nans={self.consecutive_nans}", flush=True)
            if self.consecutive_nans <= 0:
                self.interruption_scheduled = True
        else:
            self.consecutive_nans = self.default_consecutive_nans
    def on_epoch_end(self, state, epoch, **_):
        if self.interruption_scheduled:
            state["comment"] = f"NaNs@epoch{epoch}"

class GatherCycleMetrics(Callback):
    before = ["StateLogger"]
    excluded_keys = ["cycle_name", "cycle_type", "epoch", "mode"]
    def on_epoch_begin(self, **_):
        self.acc = {}
    def on_cycle_end(self, cycle_name, state, **_): # 'state' argument doesn't contain itself
        for name in state:
            if name not in self.excluded_keys:
                self.acc[cycle_name + "_" + name] = state[name]
    def on_epoch_end(self, state, **_): # 'state' argument in R/W
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
    before = ["StateLogger"]
    def init(self, exp):
        self.learning_rate_getter = exp.get_learning_rate
    def on_epoch_end(self, state, **_):
        state["learning_rate"] = self.learning_rate_getter()

@dataclass
class FindLearningRate(Callback):
    before = ["GatherCycleMetrics"]
    initial_learning_rate: float = 10e-10
    final_learning_rate: float = 10e1
    steps: int = 1000
    def __post_init__(self):
        self.history = []
        self.interruption_scheduled = False
    def init(self, exp):
        self.learning_rate_setter = exp.set_learning_rate
        self.learning_rate_getter = exp.get_learning_rate
        self.batch_count = sum([len(subset.keys)//exp.batch_size for subset in exp.subsets if subset.type & SubsetType.TRAIN])
        print("batch_count=", self.batch_count)
        assert all([not isinstance(cb, (LearningRateWarmUp, LearningRateDecay)) for cb in exp.cfg['callbacks']]), f"{self.__class__} is incompatible with existing callbacks"
    def compute_factor(self, step):
        initial_exp = np.log(self.initial_learning_rate)
        final_exp = np.log(self.final_learning_rate)
        exp = initial_exp + (final_exp - initial_exp)*step/self.steps
        return np.exp(exp)
    def on_batch_begin(self, cycle_type, epoch, batch, **_):
        if cycle_type == SubsetType.TRAIN:
            step = epoch*self.batch_count + batch
            if step >= self.steps:
                self.interruption_scheduled = True
            factor = self.compute_factor(step=step)
            self.learning_rate_setter(self.initial_learning_rate * factor)
    def on_batch_end(self, cycle_type, loss, **_):
        if cycle_type == SubsetType.TRAIN:
            loss = np.mean(loss)
            self.history.append(loss)
            if len(self.history) == self.steps:
                pickle.dump({self.compute_factor(step): loss for step, loss in enumerate(self.history)}, open("loss_history.pickle", "wb"))
                raise
    def on_cycle_end(self, state, **_):
        state["loss_evolution"] = np.array(self.history)
    def on_epoch_begin(self, **_):
        if self.interruption_scheduled:
            raise FailedTrainingError


@dataclass
class LearningRateDecay(Callback):
    """ Decay learning rate by a given factor spread on every batch for a given
        number of epochs.
        This callback is incompatible with a `LearningRateWarmUp` callback
        because learning rate is set from both callbacks at the beginning of
        each batch.
        start and duration are expressed in epochs here
    """
    start: Tuple[int, list, tuple]
    duration: int = 1
    factor: float = 0.1
    def __post_init__(self):
        assert isinstance(self.start, (list, tuple, range, type(None))), "start must be a list or range or None"
        self.start = [] if self.start is None else list(self.start)
    def init(self, exp):
        self.learning_rate_setter = exp.set_learning_rate
        self.learning_rate_getter = exp.get_learning_rate
        self.learning_rate = self.learning_rate_getter()
        self.batch_count = sum([len(subset.keys)//exp.batch_size for subset in exp.subsets if subset.type & SubsetType.TRAIN])
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
        self.batch_count = sum([len(subset.keys)//exp.batch_size for subset in exp.subsets if subset.type & SubsetType.TRAIN])
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
    period: int = 10
    def __post_init__(self):
        self.tic = time.time()
    def on_batch_end(self, epoch, batch, loss, **_):
        tac = time.time()
        if tac - self.tic > self.period:
            print(f"Epoch:{epoch} Batch:{batch} - batch loss={loss}")
            self.tic = tac
