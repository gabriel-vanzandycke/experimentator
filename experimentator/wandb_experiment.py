import wandb
import json
import numpy as np
from mlworkflow import lazyproperty
from .base_experiment import BaseExperiment
from .callbacks import Callback

class WandBExperiment(BaseExperiment):
    @lazyproperty
    def wandb(self):
        wandb.init(project=self.cfg.get("project_name", "none"), reinit=True, config=self.cfg)
        wandb.run.name = json.dumps(self.grid_sample)
        return wandb

class LogStateWandB(Callback):
    precedence = 100 # very last
    def __init__(self, exp): # pylint: disable=super-init-not-called
        self.wandb = exp.wandb
    def on_epoch_begin(self, **_):
        self.state = {}
    def on_cycle_end(self, subset_name, state, **_):
        self.state.update({subset_name + "_" + k: v for k,v in state.items()})
    def on_epoch_end(self, **_):
        self.wandb.log(self.state) # log *once* per epoch

