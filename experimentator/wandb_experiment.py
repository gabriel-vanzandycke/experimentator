import os
import pandas
from mlworkflow import lazyproperty
import wandb
from .logger_experiment import LoggerCallback

os.environ["WANDB_SILENT"] = "true"

class LogStateWandB(LoggerCallback):
    @lazyproperty
    def wandb(self):
        wandb.init(
            project=self.project_name,
            reinit=True,
            config=self.config,
            settings=wandb.Settings(show_emoji=False, show_info=False, show_warnings=False)
        )
        wandb.run.name = self.run_name
        return wandb
    def __del__(self):
        self.wandb.finish()
    def on_epoch_end(self, **_):
        report = {}
        for key, data in self.state.items():
            if isinstance(data, pandas.DataFrame):
                report[key] = self.wandb.Table(dataframe=data)
            else:
                report[key] = data
        self.wandb.log(report) # log *once* per epoch
