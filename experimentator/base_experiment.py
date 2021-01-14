import os
import traceback
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from tqdm import tqdm, tqdm_notebook
from ipywidgets import Output

from mlworkflow import SideRunner, LazyConfigurable, lazyproperty, TransformedDataset, PickledDataset, pickle_or_load

from .utils import DataCollector, StagnationError, get_axes, build_metrics_axes, plot, datetime, transforms_to_name, mkdir, find, RobustBatchesDataset


class BaseExperiment(LazyConfigurable):
    batch_count = 0

    def __del__(self):
        pass

    def get(self, key, default):
        return self.cfg.get(key, default)

    @property
    def epoch(self):
        return 0 # can be overwritten in a LoggedExperiment to continue a loaded traning

    @lazyproperty
    def grid_sample(self):
        return dict(self.cfg.get("grid_sample", {}))

    @lazyproperty
    def dataset(self):
        dataset_name = self["dataset_name"]
        dataset = PickledDataset(find(dataset_name, verbose=False))
        dataset_folder = os.environ["SSD_DATASETS_FOLDER"]

        early_transforms = [t() for t in self.get("early_transforms", [])]
        late_transforms = [t() for t in self.get("late_transforms", [])]
        if early_transforms:
            dataset = TransformedDataset(dataset, early_transforms)
            filename = os.path.join(dataset_folder, dataset_name[:-7], transforms_to_name(early_transforms))
            mkdir(os.path.dirname(filename))
            dataset = pickle_or_load(dataset, filename)
        if late_transforms:
            dataset = TransformedDataset(dataset, late_transforms)
        dataset = RobustBatchesDataset(dataset)
        return dataset

    def subsets_init(self):
        keys_splitter = self["keys_splitter"]()
        return keys_splitter(self.dataset.keys, fold=self.get("fold", default=0))

    @lazyproperty
    def subsets(self):
        return self.subsets_init()

    def progress(self, generator, **kwargs):
        return tqdm(generator, **kwargs, disable=self.get("hide_progress", False), leave=False)

    @property
    def batch_size(self):
        return self["batch_size"]

    def batch_generator(self, keys, batch_size=None):
        batch_size = batch_size or self.batch_size
        # TODO handle balanced batches
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
        for subset in self.subsets:
            assert subset.keys, "Empty subset is not allowed: {}".format(subset.name)
            if subset.do_run_epoch(epoch):
                self.run_cycle(subset, progress)
        progress.close()

    def train(self, epochs):
        self.init_model() # TODO: move elsewhere
        range_epochs = range(self.epochs+1, epochs+1)
        for epoch in self.progress(range_epochs, desc="epochs"):
            self.run_epoch(epoch)

    def predict(self, data):
        return self.batch_infer(data)

class CallbackedExperiment(BaseExperiment): # pylint: disable=abstract-method
    state = {}

    @lazyproperty
    def callbacks(self):
        callbacks = [cb(exp=self) for cb in self["callbacks"] if cb is not None]
        return sorted(callbacks, key=lambda cb: cb.precedence)

    def fire(self, event):
        for cb in self.callbacks:
            cb.fire(event, self.state)

    def run_batch(self, data, *args, **kwargs):
        self.state["batch_id"] = self.state["batch_id"] + 1
        self.state["data"] = data
        self.fire("batch_begin")
        result = super().run_batch(data=data, *args, **kwargs)
        self.state.update(**{k:v for k,v in result.items() if k in list(self.metrics.keys())+["loss"]})
        del self.state["data"]
        self.fire("batch_end")
        return result

    def run_cycle(self, subset, *args, **kwargs):
        self.state["batch_id"] = 0
        self.state["subset_name"] = subset.name
        self.fire("cycle_begin")
        super().run_cycle(subset, *args, **kwargs)
        self.fire("cycle_end")

    def run_epoch(self, *args, **kwargs):
        self.state["epoch"] = self.state.get("epoch", self.epoch)# + 1 # why +1 ?
        self.fire("epoch_begin")
        super().run_epoch(*args, **kwargs)
        self.fire("epoch_end")

class AsyncExperiment(BaseExperiment): # pylint: disable=abstract-method
    @lazyproperty
    def side_runner(self):
        return SideRunner()

    def batch_generator(self, *args, **kwargs):
        batch_generator = super().batch_generator(*args, **kwargs)
        return self.side_runner.yield_async(batch_generator)

class LoggedExperiment(BaseExperiment):
    @lazyproperty
    def logger(self):
        self.cfg["experiment_id"] = datetime()["dt"]
        filename = self["logger_filename"].format(self.cfg["experiment_id"])
        logger = DataCollector(filename, external=["weights"])
        logger["cfg"] = self.cfg
        return logger

    @lazyproperty
    def subsets(self):
        # Note: cannot use dict.set_default(key, fun()) because fun() would be called anyway
        if "subsets" not in self.logger:
            self.logger["subsets"] = self.subsets_init()
        return self.logger["subsets"]

    @property
    def weights(self):
        raise NotImplementedError("Should be implemented in the framework specific Experiment.")

    @property
    def outputs(self):
        raise NotImplementedError("Should be implemented in the task specific Experiment")

    def run_epoch(self, epoch):
        super().run_epoch(epoch)

        # Log data
        self.logger["learning_rate"] = self.learning_rate
        self.logger["batch_size"] = self["batch_size"]
        # Save weights
        self.logger["weights"] = self.weights
        # Checkpoint logs
        self.logger.checkpoint()
        # Prevent stagnation
        loss_history = np.array([metrics["loss"] for metrics in self.logger["training_metrics", :]])
        StagnationError.catch(loss_history)

    def metric_history(self, metric_name, subset_name):
        return np.array([metrics[metric_name] for metrics in self.logger["{}_metrics".format(subset_name), :] if metrics])


    @property
    def epochs(self):
        # "batch_size" being one of the last element logged on the logger
        return len(self.logger["batch_size",:])

    def train(self, *args, **kwargs):
        try:
            super().train(*args, **kwargs)
            self.logger["message"] = "Training successfull"
            self.logger.checkpoint()
        except KeyboardInterrupt as e:
            self.logger["message"] = "Interrupted by KeyboardInterrupt"
            self.logger.checkpoint()
            raise e
        except Exception as e: # pylint: disable=broad-except
            self.logger["message"] = str(e)
            self.logger["backtrace"] = traceback.format_exc()
            self.logger.checkpoint()
            raise e

class NotebookExperiment(LoggedExperiment, LazyConfigurable):  # pylint: disable=abstract-method
    @lazyproperty
    def tqdm_output(self):
        output = Output()
        display.display(output)
        return output

    @lazyproperty
    def plot_output(self):
        output = Output()
        display.display(output)
        return output

    @lazyproperty
    def draw_output(self):
        output = Output()
        display.display(output)
        return output

    def __del__(self):
        self.plot_output.clear_output()
        self.tqdm_output.clear_output()
        self.draw_output.clear_output()
        super().__del__()

    def progress(self, generator, **kwargs):
        with self.tqdm_output:
            return tqdm_notebook(generator, **kwargs, disable=self.get("hide_progress", False), leave=False)

    def plot_metrics(self, metrics_names=None, subsets_names=None, fig=None, figsize=None):
        subsets_names = subsets_names or [s.name for s in self.subsets]
        metrics_names = metrics_names or [m for m in self.logger["metrics"] if m not in ["subset_name"]]#["accuracy", "batch_time", "precision", "recall", "loss"]#self.logger["metrics"]
        axes = build_metrics_axes(metrics_names, figsize) if fig is None else fig.get_axes()
        for subset in [s for s in self.subsets if s.name in subsets_names]:
            for idx, metric_name in enumerate(metrics_names):
                ax = axes[idx]
                try:
                    data = self.metric_history(metric_name, subset.name)
                except KeyError:
                    continue
                plot(ax, data, label=subset.legend, legend=True, average=False, linestyle=subset.linestyle, marker=".", linewidth=1, markersize=10)
                if np.any(data):
                    if metric_name == "loss" and subset.name == "training" and data.shape[0] > 3:
                        avg = np.nanmean(data[1:])
                        ax.set_ylim([0, 2*avg])
                    if "time" in metric_name and subset.name == "training" and not np.isnan(data).any():
                        ax.set_ylim([0, 11*np.nanmax(data)/10])

        return axes[0].figure

    def illustrate(self, size=8, **kwargs):
        keys = [subset.keys[0] for subset in self.subsets]
        axes = get_axes(cols=len(keys), size=size, squeeze=True)
        # TODO: automatically get subset
        for ax, key, subset in zip(axes, keys, self.subsets):
            input_data = {"batch_"+name: value for name, value in self.dataset.query([key]).items()}
            output_data = self.predict(input_data)
            input_data = {name: value[0] for name, value in input_data.items()}
            output_data = {name: value[0] for name, value in output_data.items()}
            image = self.illustrator(input_data, output_data, **kwargs)
            ax.imshow(image)
            ax.set_title("{} sample".format(subset.name if subset else "none"))
            ax.set_axis_off()
        return axes[0].figure

    def run_epoch(self, *args, **kwargs):
        super().run_epoch(*args, **kwargs)

        with self.plot_output:
            display.clear_output(wait=True)
            display.display(self.plot_metrics())
            plt.close()

        with self.draw_output:
            display.clear_output(wait=True)
            display.display(self.illustrate())
            plt.close()

class LearningRate():
    def __init__(self, learning_rate_base, decay_factor=2, decay_period=30):
        self.learning_rate_base = learning_rate_base
        self.decay_factor = decay_factor
        self.decay_period = decay_period

    def __call__(self, epoch, cost_history):
        # TODO: implement decay strategy
        return self.learning_rate_base
