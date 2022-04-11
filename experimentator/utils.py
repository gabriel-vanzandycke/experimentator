import errno
import inspect
import multiprocessing.pool
import os
import random
import re
import sys

import datetime as DT
import dataclasses
from enum import IntFlag

import dill as pickle
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from mlworkflow import get_callable, Dataset
from mlworkflow.datasets import batchify

def insert_suffix(filename, suffix):
    root, ext = os.path.splitext(filename)
    return root + suffix + ext


class ExperimentMode(IntFlag):
    NONE  = 0
    TRAIN = 1
    EVAL  = 2
    INFER = 4
    ALL   = -1

# TODO: could be implemented using dataclass
class ChunkProcessor():
    mode = ExperimentMode.TRAIN | ExperimentMode.EVAL | ExperimentMode.INFER
    def __repr__(self):
        try:
            config = getattr(self, "config", {name: getattr(self, name) for name in inspect.getfullargspec(self.__init__).args[1:]})
        except KeyError as e:
            print("You should implement the 'config' property that returns a dictionnary of config given to '__init__'")
            raise e
        attributes = ",".join("{}={}".format(k, v) for k,v in config.items())
        return "{}({})".format(self.__class__.__name__, attributes)

class OutputInhibitor():
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        if self.name:
            print("Launching {}... ".format(self.name), end="")
        self.ps1, self.ps2 = getattr(sys, "ps1", None), getattr(sys, "ps2", None)
        if self.ps1:
            del sys.ps1
        if self.ps2:
            del sys.ps2
        self.stderr = sys.stderr
        self.fp = open(os.devnull, "w")
        sys.stderr = self.fp
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ps1:
            sys.ps1 = self.ps1
        if self.ps2:
            sys.ps2 = self.ps2
        sys.stderr = self.stderr
        self.fp.close()
        if self.name:
            print("Done.")


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


class StagnationError(Exception):
    @classmethod
    def catch(cls, history, tail=5):
        epoch = len(history)
        history = [h for h in history if h!=np.nan]
        if len(history) > tail and len(set(history[-tail:])) == 1:
            raise cls("after {} epochs.".format(epoch))


def find(filename, dirs=None, verbose=True):
    if os.path.isabs(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        return filename

    dirs = dirs or [os.getcwd(), *os.getenv("DATA_PATH").split(":")]
    for path in dirs:
        if path is None:
            continue
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            if verbose:
                print("{} found in {}".format(filename, filepath))
            return filepath

    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                "{} (searched in {})".format(filename, dirs))

def datetime():
    dt = DT.datetime.now()
    d = dt.strftime("%Y%m%d")
    t = dt.strftime("%H%M%S")
    dt = "{}_{}".format(d, t)
    return {"d": d, "t": t, "dt": dt}


PLOT_HEIGHT = 4
def build_metrics_axes(fetches, figsize=None):
    figsize = figsize if figsize else (20, PLOT_HEIGHT*len(fetches))
    _, axes = plt.subplots(len(fetches), 1, figsize=figsize, squeeze=False)
    axes = [ax[0] for ax in axes]
    # axes = [axes[0] for axes in fig.subplots(len(fetches), 1, squeeze=False)]
    for ax, fetch in zip(axes, fetches):
        ax.set_title(fetch)
        ax.grid(color='lightgray', linestyle='-', linewidth=1)
        if "accuracy" in fetch or fetch in ["recall", "accuracy", "precision"]:
            ax.set_ylim([0, 1])
        elif "learning_rate" in fetch:
            ax.set_yscale('log')
    return axes

def get_axes(rows=1, cols=1, expected_shape=(1, 1), size=8, squeeze=False):
    expected_width, expected_height = expected_shape
    figsize = (size*cols, size*rows*expected_width/expected_height)
    fig = mpl.figure.Figure(figsize=figsize)
    mpl.backends.backend_agg.FigureCanvasAgg(fig)
    fig.subplots_adjust(wspace=0.05)
    fig.subplots(rows, cols)
    return fig.subplots(rows, cols, squeeze=squeeze)

def plot(ax, ydata, label, legend, replace=False, average=False, **kwargs):
    if replace:
        # Clean replaced elements
        for elem in [e for e in ax.lines+ax.collections if e.get_label() == label]:
            elem.remove()

    dim = len(ydata.shape)
    if dim == 1:
        xdata = np.arange(ydata.shape[0])
        mask = np.isfinite(ydata)
        ax.plot(xdata[mask], ydata[mask], **kwargs, label=label)
    elif dim == 2:
        xdata = np.arange(ydata.shape[1])
        mask = np.any(np.isfinite(ydata), axis=0)
        if average:
            mean = np.nanmean(ydata[:,mask], axis=0)
            var = np.sqrt(np.nanvar(ydata[:,mask], axis=0))*1
            ax.plot(xdata[mask], mean, **kwargs, label=label)
            ax.fill_between(xdata[mask], mean+var, mean-var, alpha=0.3, label=label)
        else:
            for i in range(ydata.shape[0]):
                ax.plot(xdata[mask], ydata[i, mask], **kwargs, label=label)
    else:
        raise ValueError("Invalid dimension for ydata")
    if legend:
        ax.legend()

    # ax.relim()
    # ax.autoscale_view(True, True, True)

class Callable():
    def __init__(self, callee, *args, **kwargs):
        self.callee = get_callable(callee)
        self.kwargs = kwargs
        self.args = args

    def __call__(self, *args, **kwargs):
        return self.callee(*args, *self.args, **{**kwargs, **self.kwargs})

    def __repr__(self):
        return "{}({},{})".format(self.callee, self.args, self.kwargs)

def transforms_to_name(transforms: list):
    for t in transforms:
        assert not re.match(".*object at 0x[0-9a-fA-F]*.*", str(t)), \
            "The __str__ function of {} is not implemented".format(
                t.__class__)
    # sanitize name constructed by stringification of the transforms
    return str(transforms).translate(str.maketrans({"$":  r"\$", " ": ""}))+".pickle"

linestyles = {
    "training": "--",
    "validation": "-",
    "testing": "-."
}

# ----------------------------------------------------------------------
# From: https://stackoverflow.com/a/53180921/1782553
# ----------------------------------------------------------------------
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False
    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)
# ----------------------------------------------------------------------

class DataCollector(dict):
    external = frozenset()
    def __init__(self, filename=None, readonly=False, external=None):
        super().__init__()
        assert not external or filename, "'external' argument cannot be used if no filename is specified."
        self.history = []
        self.tmp_dict = dict()
        assert isinstance(external, (list, tuple, type(None))), "argument 'external' must be a list of strings"
        self.external = set(external) if external is not None else None
        self.file = None
        self.pickler = None
        self.filename = self.format_filename(filename)
        self.directory = self.filename[:-4] if self.filename else None
        self.file_flag = "w"
        if self.filename and os.path.isfile(self.filename):
            self.file_flag = "r" if readonly else "a"
            self.file = open(self.filename, self.file_flag+"+b")
            self.file.seek(0)
            try:
                unpickler = pickle.Unpickler(self.file)
                while True:
                    data = unpickler.load()
                    self.history.append(data)
            except EOFError:
                if self.history:
                    for k,v in self.history[-1].items():
                        self.__setitem__(k,v, skip_external=True)
            self.file.close()
            self.file = None

    def __del__(self):
        if self.file:
            self.file.close()

    @classmethod
    def format_filename(cls, filename):
        if filename is None:
            return None
        assert filename[-4:] == '.dcp', "DataCollector files must end with '.dcp'. Received {}".format(filename)
        datetime_ = DT.datetime.now()
        d = datetime_.strftime("%Y%m%d")
        t = datetime_.strftime("%H%M%S")
        dt = "{}_{}".format(d, t)
        return filename.format(dt, dt=dt, d=d, t=t)

    def save_external(self, key, value):
        mkdir(self.directory) # trim '.dcp' extension
        filename = "{}/{}_{}.data".format(self.directory, len(self.history), key)
        with open(filename, "wb") as fd:
            pickle.dump(value, fd)
        return filename

    def load_external(self, filename):
        if not isinstance(filename, str): # handle retro-compatibilty
            return filename
        with open(filename, "rb") as fd:
            value = pickle.load(fd)
        return value

    def __len__(self):
        return len(self.history) + (1 if self.tmp_dict else 0)

    def __setitem__(self, key, value, skip_external=False):
        if self.external and key in self.external and skip_external is False:
            value = self.save_external(key, value)
        super().__setitem__(key, value)
        self.tmp_dict[key] = value

    def __getitem__(self, key, skip_external=False):
        get = lambda k,v: v if not self.external or k not in self.external or skip_external else self.load_external(v)
        if isinstance(key, tuple):
            key, s = key
            history = self.history + [self.tmp_dict] if key in self.tmp_dict else self.history
            if isinstance(s, (int, np.int64)):
                return get(key, history[s][key])
            if isinstance(s, slice):
                return [get(key, data.get(key, None)) for data in history[s]]
            raise ValueError("Expected (key,slice). Received ({},{}).".format(key,s))
        if isinstance(key, int):
            it = key
            return dict((key, self[key,it]) for key in self.history[it].keys())
        if isinstance(key, slice):
            s = key
            return [dict((key, get(key, value)) for key,value in d.items()) for d in self.history[s]]
        return get(key, super().__getitem__(key))

    def setdefault(self, key, default=None):
        if key in self.tmp_dict:
            return self.tmp_dict[key]
        self[key] = default
        return default

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.tmp_dict[key]

    def pop(self, key, skip_external=False):
        value = self.__getitem__(key, skip_external=skip_external)
        del self[key]
        return value

    def update(self, **kwargs):
        for k,v in kwargs.items():
            self[k] = v

    def checkpoint(self, *keys):
        keys = keys or list(self.keys())
        checkpoint = {key: self.pop(key, skip_external=True) for key in keys}
        self.history.append(checkpoint)
        if self.filename:
            # Open file if not done already
            if not self.file:
                self.file = open(self.filename, self.file_flag+"+b")
            # Create pickler if not done already
            if not self.pickler:
                self.pickler = pickle.Pickler(self.file)
            # Dump checkpoint
            self.pickler.dump(checkpoint)
            self.file.flush()
