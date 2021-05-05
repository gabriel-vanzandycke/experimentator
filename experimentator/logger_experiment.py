import abc
import json
import os
import types
from mlworkflow import lazyproperty
from .callbacked_experiment import Callback
from .utils import DataCollector


class SaveWeights(Callback):
    precedence = 100
    min_loss = None
    def __init__(self, strategy="best"):
        self.strategy = strategy
    def init(self, exp):
        self.exp = exp
    def on_epoch_end(self, loss, epoch, **_):
        if     (self.strategy == "best" and (self.min_loss is None or loss < self.min_loss)) \
            or (self.strategy == "all"):
            self.min_loss = loss
            self.exp.save_weights(f"{self.exp.folder}/{epoch:04d}{self.exp.weights_suffix}")


class LoggerCallback(Callback, metaclass=abc.ABCMeta):
    precedence = 100 # very last
    state = {}
    def init(self, exp):
        self.project_name = exp.get("project_name", "unknown_project")
        self.run_name = json.dumps(exp.grid_sample)
        self.config = {k:str(v) for k,v in exp.cfg.items() if not isinstance(v, types.ModuleType)} # list and dictionnaries don't get printed correctly

        grid_sample = dict(exp.grid_sample) # copies the original dictionary
        grid_sample.pop("fold", None)       # removes 'fold' to be able to group runs
        self.config["group"] = grid_sample
    def on_epoch_begin(self, **_):
        self.state = {}
    def on_cycle_end(self, cycle_name, state, **_):
        excluded_keys = ["cycle_name", "cycle_type", "batch", "epoch"]
        self.state.update({cycle_name + "_" + k: v for k,v in state.items() if k not in excluded_keys})
    @abc.abstractmethod
    def on_epoch_end(self, **_):
        raise NotImplementedError()

class LogStateDataCollector(LoggerCallback):
    @lazyproperty
    def logger(self):
        filename = os.path.join(self.config.get("folder", "."), "history.dcp")
        return DataCollector(filename)
    def on_epoch_end(self, **_):
        self.logger.update(**self.state)
        self.logger.checkpoint()
















# class LoggedExperiment(BaseExperiment):
#     @lazyproperty
#     def logger(self):
#         self.cfg["experiment_id"] = datetime()["dt"]
#         filename = self.cfg.get("logger_filename", f"{self.cfg['filename']}_{{}}.dcp").format(self.cfg["experiment_id"])
#         logger = DataCollector(filename, external=["weights"])
#         logger["cfg"] = self.cfg
#         return logger

#     @property
#     def weights(self):
#         raise NotImplementedError("Should be implemented in the framework specific Experiment.")

#     @property
#     def outputs(self):
#         raise NotImplementedError("Should be implemented in the task specific Experiment")

#     def run_epoch(self, epoch):
#         super().run_epoch(epoch)

#         # Save weights
#         if self.get("save_weights", "none") == "all":
#             self.logger["weights"] = self.weights
#         self.logger["subset_names"] = list(self.subsets.keys())
#         # Checkpoint logs
#         self.logger.checkpoint()
#         # Prevent stagnation
#         loss_history = np.array([metrics["loss"] for metrics in self.logger["training_metrics", :]])
#         StagnationError.catch(loss_history)

#     def metric_history(self, metric_name, subset_name):
#         return np.array([metrics[metric_name] for metrics in self.logger["{}_metrics".format(subset_name), :] if metrics])


#     @property
#     def epochs(self):
#         # "batch_size" being one of the last element logged on the logger
#         return len(self.logger["batch_size",:])

#     def train(self, *args, **kwargs):
#         try:
#             super().train(*args, **kwargs)
#             self.logger["message"] = "Training successfull"
#             self.logger.checkpoint()
#         except KeyboardInterrupt as e:
#             self.logger["message"] = "Interrupted by KeyboardInterrupt"
#             self.logger.checkpoint()
#             raise e
#         except Exception as e: # pylint: disable=broad-except
#             self.logger["message"] = str(e)
#             self.logger["backtrace"] = traceback.format_exc()
#             self.logger.checkpoint()
#             raise e


# class NotebookExperiment(LoggedExperiment):  # pylint: disable=abstract-method
#     @lazyproperty
#     def tqdm_output(self):
#         output = Output()
#         display.display(output)
#         return output

#     @lazyproperty
#     def plot_output(self):
#         output = Output()
#         display.display(output)
#         return output

#     @lazyproperty
#     def draw_output(self):
#         output = Output()
#         display.display(output)
#         return output

#     def __del__(self):
#         self.plot_output.clear_output()
#         self.tqdm_output.clear_output()
#         self.draw_output.clear_output()
#         super().__del__()

#     def progress(self, generator, **kwargs):
#         with self.tqdm_output:
#             return tqdm_notebook(generator, **kwargs, disable=self.get("hide_progress", False), leave=False)

#     def plot_metrics(self, metrics_names=None, subsets_names=None, fig=None, figsize=None):
#         subsets_names = subsets_names or self.logger["subset_names"]
#         metrics_names = metrics_names or [m for m in self.logger["metrics"] if m not in ["subset_name"]]#["accuracy", "batch_time", "precision", "recall", "loss"]#self.logger["metrics"]
#         axes = build_metrics_axes(metrics_names, figsize) if fig is None else fig.get_axes()
#         for subset in [s for name,s in self.subsets.items() if name in subsets_names]:
#             for idx, metric_name in enumerate(metrics_names):
#                 ax = axes[idx]
#                 try:
#                     data = self.metric_history(metric_name, subset.name)
#                 except KeyError:
#                     continue
#                 plot(ax, data, label=subset.legend, legend=True, average=False, linestyle=subset.linestyle, marker=".", linewidth=1, markersize=10)
#                 if np.any(data):
#                     if metric_name == "loss" and subset.name == "training" and data.shape[0] > 3:
#                         avg = np.nanmean(data[1:])
#                         try:
#                             ax.set_ylim([0, 2*avg])
#                         except ValueError:
#                             pass
#                     if "time" in metric_name and subset.name == "training" and not np.isnan(data).any():
#                         ax.set_ylim([0, 11*np.nanmax(data)/10])

#         return axes[0].figure
