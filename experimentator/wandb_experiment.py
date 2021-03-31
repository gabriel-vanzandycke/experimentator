import os
import json
import pandas
import logging
import wandb
from .callbacked_experiment import Callback

class LogStateWandB(Callback):
    precedence = 100 # very last
    state = {}
    def __init__(self, **d):
        self.d = d
    def init(self, exp): # pylint: disable=super-init-not-called
        os.environ["WANDB_SILENT"] = "true"
        project_name = exp.get("project_name", "unknown_project")
        grid_sample = dict(exp.grid_sample) # copies the original dictionary
        grid_sample.pop("fold", None)       # removes 'fold' to be able to group runs
        run_name = json.dumps(grid_sample)
        wandb.init(
            project=project_name,
            reinit=True,
            config={k:str(v) for k,v in exp.cfg.items()},
            settings=wandb.Settings(show_emoji=False, show_info=False, show_warnings=False)
        )
        wandb.run.name = run_name
    def __del__(self):
        wandb.finish()
    def on_epoch_begin(self, **_):
        self.state = {}
    def on_cycle_end(self, subset, state, **_):
        self.state.update({subset + "_" + k: v for k,v in state.items()})
    def on_epoch_end(self, **_):
        report = {}
        for key, data in self.state.items():
            if isinstance(data, pandas.DataFrame):
                report[key] = wandb.Table(dataframe=data)
            else:
                report[key] = data
        wandb.log(report) # log *once* per epoch
