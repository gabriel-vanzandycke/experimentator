import os
import json
import neptune
from .callbacked_experiment import Callback

class LogStateNeptune(Callback):
    precedence = 100 # very last
    def __init__(self, exp): # pylint: disable=super-init-not-called
        with open(os.path.join(os.path.expanduser("~"), ".neptune.token"), "r") as f:
            os.environ["NEPTUNE_API_TOKEN"] = f.read()

        project_name = exp.project_name
        grid_sample = dict(exp.grid_sample) # copies the original dictionary
        grid_sample.pop("fold", None)       # removes 'fold' to be able to group runs
        run_name = json.dumps(grid_sample)
        neptune.init(project_qualified_name=os.path.join("gva", project_name).replace("_","-"), api_token=os.environ["NEPTUNE_API_TOKEN"])
        neptune.create_experiment(name=run_name, params=exp.cfg)
    def on_epoch_begin(self, **_):
        self.state = {}
    def on_cycle_end(self, subset, state, **_):
        self.state.update({subset + "_" + k: v for k,v in state.items()})
    def on_epoch_end(self, **_):
        for key, value in self.state.items():
            try:
                neptune.log_metric(key, value)
            except:
                print(key)

