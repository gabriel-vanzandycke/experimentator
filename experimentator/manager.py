import os
import sys
import copy
import glob
import json
import shutil
import itertools
import configparser
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook, tqdm
import numpy as np
from ast import literal_eval as make_tuple

from ipywidgets import interact, widgets, Layout, Output, interactive, Box
from IPython import display

from mlworkflow import find_files, get_callable, exec_dict, LazyConfigurable, lazyproperty

from .utils import datetime, Callable, DataCollector, mkdir

from .base_experiment import NotebookExperiment

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class Config(configparser.ConfigParser):
    def __init__(self, filename):
        super().__init__()
        self.optionxform = lambda option: option # handle upper case config keys
        self.read_string(open(filename, "r").read())
    
    def write(self, filename):
        super().write(open(filename, "w") )

class ExperimentManager(LazyConfigurable):
    def __init__(self, filename, gpu, robust=False, name=None):
        self.__dict__.pop("experiments", None)
        self.robust = robust
        config = Config(filename)
        config["Pre"]["GPU"] = "'{}'".format(gpu)
        if "Meta" in config.sections():
            self.id = config["Meta"]["id"]
            self.filename = filename
        else:
            self.id = datetime()["dt"]
            config["Meta"] = {"id": self.id, "version": 2}
            self.filename = "{}_{}.ini".format(name or os.path.basename(filename)[:-4], self.id)
            config.write(self.filename)

    @lazyproperty
    def experiments(self):
        return list(self.yield_experiments())

    def yield_experiments(self, cfgs=None):
        config = Config(self.filename)
        for cfg in (cfgs or parse_config(config, dict(Callable=Callable), self.id)):
            yield self.build_experiment(cfg)

    def execute(self, epochs):
        for exp in self.progress(self.yield_experiments()):
            config = Config(self.filename)
            grid_sample_str = json.dumps(exp.grid_sample)
            if grid_sample_str in {v:k for k,v in config["Meta"].items()}:
                print("Skipping {}".format(grid_sample_str))
                continue
            print("Doing {}".format(grid_sample_str))
            self.selected_exps = [exp]
            config["Meta"][exp.logger.filename] = grid_sample_str
            config.write(self.filename)
            try:
                exp.train(epochs)
            except KeyboardInterrupt:
                print("Interrupted by keyboard")
                break
            except Exception as e: # pylint: disable=broad-except
                if not self.robust:
                    ### should not delete experiment from "Meta" section because it can have failed at the end.
                    ### BTW, there's a bug because it seems to delate another entry
                    # config = Config(self.filename)
                    # del config["Meta"][exp.logger.filename]
                    # config.write(self.filename)
                    raise e
                print(e)

    @staticmethod
    def progress(generator, **kwargs):
        return tqdm(generator, leave=False, **kwargs)

    def build_experiment(self, cfg=None, filename=None):
        assert not cfg or not filename
        cfg = cfg or DataCollector(filename)["cfg"]
        cfg["GPU"] = "{}".format(self.gpu)
        if filename:
            cfg["logger_filename"] = filename
        return type("Exp", tuple(get_callable(n) for n in cfg["experiment"][::-1]), {})(cfg)

    # def load(self):
    #     questions = [inquirer.List('filename', message="Select process file", choices=list(reversed(find_files("process_[0-9_]*.pickle"))))]
    #     filename = inquirer.prompt(questions)['filename']
    #     cfgs = pickle.load(open(filename, "rb"))cfg
    #     options = {str(cfg["grid_sample"]): cfg for cfg in cfgs}
    #     questions = [inquirer.List('runall', message="Wich experiments would you like to execute?", choices=["all", "select"])]
    #     answer = inquirer.prompt(questions)["runall"]
    #     if answer != "all":
    #         questions = [inquirer.Checkbox('experiments', message="Select experiments", choices=list(options.keys()))]
    #         answers = inquirer.prompt(questions)['experiments']
    #     else:
    #         answers = list(options.keys())
    #     self.cfgs = [options[answer] for answer in answers]

def parse_grid_sample(grid_sample_str):
    d = json.loads(grid_sample_str) if grid_sample_str[0] == '{' else dict(make_tuple(grid_sample_str))
    if "grid_sample" in d:
        d.update(**d["grid_sample"])
        del d["grid_sample"]
    return d


class ExperimentManagerNotebook(ExperimentManager):
    def __init__(self, gpu, filename=None, *args, **kwargs):
        self.gpu = gpu
        self.output = Output()
        display.display(self.output)
        if filename is not None:
            super().__init__(filename, gpu, *args, **kwargs)
        else:
            self.load()
        self.selected_exps = None

    def __del__(self):
        if self.output:
            self.output.clear_output()

    def progress(self, generator, **kwargs):
        with self.output:
            return tqdm_notebook(generator, leave=False, **kwargs)

    def build_experiment(self, cfg=None, filename=None):
        assert not cfg or not filename
        cfg = cfg if filename is None else DataCollector(filename)["cfg"]
        cfg["GPU"] = "{}".format(self.gpu)
        if filename:
            cfg["logger_filename"] = filename
        classes = tuple(get_callable(n) for n in cfg["experiment"][::-1])
        if all(["NotebookExperiment" not in str(c) for c in classes]):
            classes = (NotebookExperiment,) + classes
        return type("Exp", classes, {})(cfg)

    def load_experiments(self, **params):
        def plot_metrics(filenames):
            self.experiments = [self.build_experiment(filename=filename) for filename in filenames]
            fig = None
            for exp in self.experiments:
                if "backtrace" in exp.logger:
                    print(exp.logger["backtrace"], file=sys.stderr)
                else:
                    if "message" in exp.logger:
                        print(exp.logger["message"], file=sys.stderr)
                fig = exp.plot_metrics(fig=fig, metrics_names=exp.get("metrics_names", ["factor", "accuracy", "precision", "recall", "loss", "batch_time"]))
        config = Config(self.filename)
        options_reversed = {filename:parse_grid_sample(grid_sample_str) for filename, grid_sample_str in dict(config["Meta"]).items() if filename not in ["version", "id"]}
        options_reversed = {filename:grid_sample for filename, grid_sample in options_reversed.items() if
            all([params.get(k, "any") in [str(v), "any"] for k,v in grid_sample.items()])
        }
        options = {str(tuple(v.items())):k for k,v in options_reversed.items()}
        layout = Layout(width="95%", height="200px")
        exp_selector = interactive(plot_metrics, filenames=widgets.SelectMultiple(options=options, description="grid samples", layout=layout))
        display.display(exp_selector)

    def load(self):
        @interact(filename=reversed(find_files("*.ini")))
        def _(filename):
            super(ExperimentManagerNotebook, self).__init__(filename, self.gpu)
            params_lists = {}
            config = Config(self.filename)
            for grid_sample in [parse_grid_sample(grid_sample_str) for filename, grid_sample_str in dict(config["Meta"]).items() if filename not in ["version", "id"]]:
                for k,v in grid_sample.items():
                    params_lists.setdefault(k, set()).add(str(v))
            for k,v in params_lists.items():
                params_lists[k] = ["any"] + sorted(list(v))
            button = widgets.Button(description="Move process to trash")
            button.on_click(lambda e: self.delete_process())
            box_layout = Layout(display='flex-end', flex_flow='row-reverse', align_items='stretch')
            display.display(Box(children=[button], layout=box_layout))
            w = interactive(self.load_experiments, **params_lists)
            display.display(w)
            # box_layout = Layout(overflow='scroll hidden', flex_flow='row', display='flex')
            # container = Box(children=items, layout=box_layout)
            # VBox([Label('Scroll horizontally:'), carousel])


    def delete_process(self):
        for src in glob.glob("*/training_{}*".format(self.id)):
            dst = os.path.join(os.environ["HOME"], "trash", src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            print("'{}' moved to '{}'".format(src, dst))
        shutil.move(self.filename, os.path.join(os.environ["HOME"], "trash", self.filename))
        print("'{}' moved to '{}'".format(self.filename, os.path.join(os.environ["HOME"], "trash",self.filename)))





def delete_weights(experiments):
    acc_idxs_folds = dict()
    files_list = []
    for exp in experiments:
        exp_index = exp.loggers[0]['cfg']['logger_prefix'][-2:]
        acc_idxs_folds[exp_index]=top_n_weight(exp)
        files_list.append(exp.loggers[0]['cfg']['logger_prefix'])

    for fl in files_list:
        for f in glob.glob(fl+"*"):
            if os.path.isfile(f) == False:
                fold = int(f[-2:])
                e = f[-5:-3]
                top_weights_f = acc_idxs_folds[e][fold]
                top_weights_f = [str(f) + "/" + str(t) + "_weights.data" for t in top_weights_f]
                for w in glob.glob("{}/*".format(f)):
                    if (w in top_weights_f) == False:
                        os.remove(w)
                print("Deleted weights of " + str(f))

def delete_exps(experiments):
    files_list = []
    for exp in experiments:
        files_list.append(exp.loggers[0]['cfg']['logger_prefix'])
    for fl in files_list:
        for f in glob.glob(fl+"*"):
            os.remove(f) if os.path.isfile(f) else shutil.rmtree(f)
        print("Deleted " + str(fl))

def top_n_weight(exp, n=5):
    accuracy_v = exp.metrics_history()[('accuracy_GV', 'validation')]
    accuracy_t = exp.metrics_history()[('accuracy_GV', 'testing')]
    acc_idxs_folds = []

    for a in accuracy_v:
        acc_idxs = []
        for _ in range(n):
            if all(np.isnan(a)) == False:
                idx = np.nanargmax(a)
                acc_idxs.append(idx)
                a[idx] = np.nan
            else:
                acc_idxs.append(None)
        acc_idxs_folds.append(acc_idxs)

    for i, a in enumerate(accuracy_t):
        acc_idxs = []
        for _ in range(n):
            if all(np.isnan(a)) == False:
                idx = np.nanargmax(a)
                acc_idxs.append(idx)
                a[idx] = np.nan
            else:
                acc_idxs.append(None)
        acc_idxs_folds[i].extend(acc_idxs)
    return acc_idxs_folds

# def build_experiment(filename):
#     assert os.path.isfile(filename), "'{}' doesn't exist".format(filename)
#     logger = DataCollector(filename)
#     return ExperimentManagerNotebook.build_experiment(logger["cfg"], filename)



def parse_config(config, env, manager_id):
    def create_config(pre_section, grid_sample, post_section):
        cfg = copy.deepcopy(pre_section)
        cfg.update(grid_sample)
        exec_dict(cfg, post_section.items(), {**env, **cfg})
        logs_folder_name = cfg['dataset_name'][:-7]
        mkdir(logs_folder_name)
        logger_filename = f"{logs_folder_name}/training_{manager_id}_{{}}.dcp" # default logger_filename
        if "Meta" in config:
            d = {v:k for k,v in dict(config["Meta"]).items()}
            key = json.dumps(dict(grid_sample)) if int(config["Meta"].get("version", 1)) >= 2 else str(tuple(dict(grid_sample).items()))
            logger_filename = d.get(key, logger_filename)
        cfg.update(
            manager_id=manager_id,
            grid_sample=grid_sample,
            logger_filename=logger_filename,
        )
        return cfg

    sections = {section: dict(config[section]) for section in config.sections()}

    # Parse ["Pre"] section
    pre_section = dict()
    exec_dict(pre_section, sections["Pre"].items(), env)

    # Parse ["GridSearch"] section
    grid_cfg = dict()
    exec_dict(grid_cfg, sections.get("GridSearch", {}).items(), env)

    post_section = sections["Post"]

    kvs = [[(k, v) for v in grid_cfg[k]] for k in grid_cfg]
    grid = list(itertools.product(*kvs))
    return [create_config(pre_section, grid_sample, post_section) for grid_sample in grid]
