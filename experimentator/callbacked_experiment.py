import time
import numpy as np

from mlworkflow import lazyproperty

from .base_experiment import BaseExperiment

class CallbackedExperiment(BaseExperiment): # pylint: disable=abstract-method
    state = {}

    @lazyproperty
    def callbacks(self):
        callbacks = [cb(exp=self) for cb in self.cfg["callbacks"] if cb is not None]
        return sorted(callbacks, key=lambda cb: cb.precedence)

    def fire(self, event):
        for cb in self.callbacks:
            cb.fire(event, self.state)

    @lazyproperty
    def metrics(self):
        """ Describes network's outputs that should be saved in the current state
        """
        return {}

    def run_batch(self, *args, **kwargs):
        self.state["batch"] = self.state["batch"] + 1
        self.fire("batch_begin")
        result = super().run_batch(*args, **kwargs)
        self.state.update(**{k:v for k,v in result.items() if k in list(self.metrics.keys())+["loss"]})
        self.fire("batch_end")
        return result

    def run_cycle(self, subset, *args, **kwargs):
        self.state["subset"] = subset.name
        self.state["batch"] = 0
        self.fire("cycle_begin")
        super().run_cycle(subset, *args, **kwargs)
        self.fire("cycle_end")

    def run_epoch(self, epoch, *args, **kwargs):
        self.state["epoch"] = epoch
        self.fire("epoch_begin")
        super().run_epoch(epoch, *args, **kwargs)
        self.fire("epoch_end")


class Callback():
    precedence = 10
    def __init__(self, *args, **kwargs):
        pass # required to call constructor with 'exp'
    def fire(self, event, state):
        cb = getattr(self, "on_{}".format(event), None)
        if cb:
            cb(**state, state=state) # pylint: disable=not-callable

class InitState(Callback):
    precedence = 0 # very first
    def on_epoch_begin(self, state, **_):
        for key in [k for k in state if k!= "epoch"]:
            state[key] = np.nan

class AverageLoss(Callback):
    def on_cycle_begin(self, **_):
        self.loss = []
    def on_batch_end(self, loss, **_):
        self.loss.append(loss)
    def on_cycle_end(self, state, **_):
        state["loss"] = np.mean(self.loss)

class MeasureTime(Callback):
    precedence = 70
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
