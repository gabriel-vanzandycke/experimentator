from functools import cached_property
import os
import json

from experimentator import StateLogger, ConfusionMatrix
import pandas
import wandb

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

class LogStateWandB(StateLogger):
    best_report = {}
    def __init__(self, criterion_metric=None, higher_is_better=True, mode="online"):
        self.criterion_metric = criterion_metric
        self.mode = mode
        self.higher_is_better = higher_is_better
        self.initialized = False
    @cached_property
    def wandb_run(self):
        run = wandb.init(
            project=self.project_name,
            reinit=True,
            config=self.config,
            settings=wandb.Settings(show_emoji=False, _save_requirements=False),
            mode=self.mode,
        )
        run.name = self.run_name
        self.initialized = True
        return run
    def __del__(self):
        if self.initialized:
            try:
                self.wandb_run.finish()
            except RuntimeError:
                self.wandb_run.finish() # try again
    def on_epoch_end(self, state, **_):
        report = {}
        for key, data in state.items():
            if isinstance(data, pandas.DataFrame):
                report[key] = wandb.Table(dataframe=data)
            # elif isinstance(data, ConfusionMatrix):
                # TODO: https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM
                # report[key] = wandb.plot.confusion_matrix(probs=None, y_true=ground_truth, preds=predictions, class_names=class_names)})
            else:
                try:
                    json.dumps(data)
                    report[key] = data
                except TypeError: # not JSON serializable
                    continue
        self.wandb_run.log(report) # log *once* per epoch

        if self.criterion_metric and self.criterion_metric in report:
            if not self.best_report or (
                   (    self.higher_is_better and report[self.criterion_metric] > self.best_report[self.criterion_metric])
                or (not self.higher_is_better and report[self.criterion_metric] < self.best_report[self.criterion_metric])
            ):
                self.best_report = report
            self.wandb_run.summary.update(self.best_report)
