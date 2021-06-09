import os
import pandas
from mlworkflow import lazyproperty
import wandb
from experimentator import StateLogger

os.environ["WANDB_SILENT"] = "true"

class LogStateWandB(StateLogger):
    best_report = {}
    def __init__(self, criterion_metric=None):
        self.criterion_metric = criterion_metric
    @lazyproperty
    def wandb(self):
        wandb.init(
            project=self.project_name,
            reinit=True,
            config=self.config,
            settings=wandb.Settings(show_emoji=False)
        )
        wandb.run.name = self.run_name
        return wandb
    def __del__(self):
        self.wandb.finish()
    def on_epoch_end(self, state, **_):
        report = {}
        for key, data in state.items():
            if key not in self.excluded_keys:
                if isinstance(data, pandas.DataFrame):
                    report[key] = self.wandb.Table(dataframe=data)
                else:
                    report[key] = data
        self.wandb.log(report) # log *once* per epoch

        if self.criterion_metric and self.criterion_metric in report:
            if not self.best_report or report[self.criterion_metric] > self.best_report[self.criterion_metric]:
                self.best_report = report
            self.wandb.run.summary.update(self.best_report)

