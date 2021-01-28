import json
import wandb
from .callbacked_experiment import Callback

class LogStateWandB(Callback):
    precedence = 100 # very last
    def __init__(self, exp): # pylint: disable=super-init-not-called
        project_name = exp.get("project_name", "unknown_project")
        grid_sample = dict(exp.grid_sample) # copies the original dictionary
        grid_sample.pop("fold", None)       # removes 'fold' to be able to group runs
        run_name = json.dumps(grid_sample)
        wandb.init(
            project=project_name,
            reinit=True,
            config=exp.cfg,
            settings=wandb.Settings(show_emoji=False)
        )
        wandb.run.name = run_name
    def on_epoch_begin(self, **_):
        self.state = {}
    def on_cycle_end(self, subset, state, **_):
        self.state.update({subset + "_" + k: v for k,v in state.items()})
    def on_epoch_end(self, **_):
        wandb.log(self.state) # log *once* per epoch

