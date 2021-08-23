import os
import pandas
from mlworkflow import lazyproperty
from experimentator import StateLogger
import wandb

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

class LogStateWandB(StateLogger):
    best_report = {}
    def __init__(self, criterion_metric=None, mode="online"):
        self.criterion_metric = criterion_metric
        self.mode = mode
        self.initialized = False
    @lazyproperty
    def wandb_run(self):
        run = wandb.init(
            project=self.project_name,
            reinit=True,
            config=self.config,
            settings=wandb.Settings(show_emoji=False),
            mode=self.mode,
        )
        run.name = self.run_name
        self.initialized = True
        return run
    def __del__(self):
        if self.initialized
            wandb.finish()
    def on_epoch_end(self, state, **_):
        report = {}
        for key, data in state.items():
            if key not in self.excluded_keys:
                if isinstance(data, pandas.DataFrame):
                    report[key] = wandb.Table(dataframe=data)
                else:
                    report[key] = data
        self.wandb_run.log(report) # log *once* per epoch

        if self.criterion_metric and self.criterion_metric in report:
            if not self.best_report or report[self.criterion_metric] > self.best_report[self.criterion_metric]:
                self.best_report = report
            self.wandb_run.summary.update(self.best_report)

