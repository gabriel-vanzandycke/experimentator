















# class LoggedExperiment(BaseExperiment):
#     @cached_property
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
#     @cached_property
#     def tqdm_output(self):
#         output = Output()
#         display.display(output)
#         return output

#     @cached_property
#     def plot_output(self):
#         output = Output()
#         display.display(output)
#         return output

#     @cached_property
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
#         for subset in [s for s in self.subsets if s.name in subsets_names]:
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
