import argparse
import ast
import copy
import datetime
from enum import Enum
from functools import cached_property
import multiprocessing

import itertools
import logging
import os
import time

import astunparse
from mlworkflow import SideRunner
from experimentator import DummyExperiment
from .utils import find, mkdir, NestablePool
from .callbacked_experiment import FailedTrainingError
# pylint: disable=logging-fstring-interpolation, logging-format-interpolation

def product_kwargs(**kwargs):
    try:
        kvs = [[(k, v) for v in kwargs[k]] for k in kwargs]
    except BaseException as e:
        raise SyntaxError(f"Error parsing: {kwargs}") from e
    yield from [dict(kv) for kv in itertools.product(*kvs)]

def update_ast(tree, overwrite, allow_double_assignation=False, allow_tuple_assignation=False):
    met_targets = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            assert allow_tuple_assignation or isinstance(target, ast.Name), "Tuple assignation is not allowed in config files (e.g. `a,b=1,2`). Impossible to overwrite '{}' of type '{}'".format(target.id, type(target))
            assert allow_double_assignation or target.id not in met_targets, "Double assignation is not allowed in config files. '{}' seems to be assigned twice.".format(target.id)
            if target.id in overwrite:
                node.value = ast.parse(repr(overwrite.pop(target.id))).body[0].value
                met_targets.append(target.id)
    # Add remaining keys
    for key, value in overwrite.items():
        tree.body.append(ast.Assign([ast.Name(id=key, ctx=ast.Store())], ast.Constant(value, kind=None)))
    ast.fix_missing_locations(tree)
    return overwrite

def parse_config_str(config_str):
    config = {}
    exec(config_str, None, config) # pylint: disable=exec-used
    return config

def parse_config_file(config_filename):
    with open(config_filename, "r") as f:
        config = parse_config_str(f.read())
    return config

def get_worker_id(*_):
    time.sleep(.1)
    return os.getpid()

def set_cuda_visible_device(index):
    time.sleep(.1)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(',')[index]

def build_experiment(config_filename, **kwargs):

    with open(find(config_filename)) as f:
        tree = ast.parse(f.read())
    update_ast(tree, dict({**kwargs, "filename": config_filename}), allow_double_assignation=True)
    config_str = astunparse.unparse(tree)

    config = parse_config_str(config_str)
    return type("Exp", tuple(config["experiment_type"][::-1]), {})(config)

class JobStatus(Enum):
    TODO = 0
    BUSY = 1
    FAIL = 2
    DONE = 3

class Job():
    def __init__(self, filename, config_tree, dummy=False, grid_sample=None):
        self.filename = filename
        self.dummy = dummy
        self.config_tree = config_tree
        self.grid_sample = grid_sample or {}
        self.status = JobStatus.TODO

    def update_ast(self, **kwargs):
        return update_ast(self.config_tree, dict(kwargs)) # dict() makes a copy

    @property
    def config_str(self):
        return astunparse.unparse(self.config_tree)

    @cached_property
    def config(self):
        config = {}
        exec(self.config_str, None, config) # pylint: disable=exec-used
        return {**config, "grid_sample": self.grid_sample, "filename": self.filename}

    @cached_property
    def exp(self):
        if self.dummy:
            self.config["experiment_type"].append(DummyExperiment)
        return type("Exp", tuple(self.config["experiment_type"][::-1]), {})(self.config)

    def run(self, epochs, keep=True, worker_ids=None, **runtime_cfg):
        project_name = runtime_cfg.get("project_name", os.path.splitext(os.path.basename(self.filename))[0])
        experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        worker_index = worker_ids[get_worker_id()] if worker_ids else 0

        # Update config tree with runtime config
        unoverwritten = self.update_ast(**runtime_cfg, epochs=epochs)
        if unoverwritten:
            logging.warning("Un-overwritten runtime kwargs: {}".format(list(unoverwritten.keys())))

        # Write config string to file
        folder = os.path.join(os.getenv("RESULTS_FOLDER", "."), project_name, experiment_id)
        mkdir(folder)
        filename = os.path.join(folder, "config.py")
        with open(filename, "w") as f:
            f.write(self.config_str)

        # Add run and project names
        self.config.update(project_name=project_name, experiment_id=experiment_id, worker_index=worker_index, folder=folder, dummy=self.dummy)

        # Launch training
        try:
            self.status = JobStatus.BUSY
            self.exp.logger.info(f"{project_name}.{experiment_id} doing {self.grid_sample}")
            self.exp.train(epochs=epochs)
        except FailedTrainingError as e:
            self.status = JobStatus.FAIL
            self.exp.logger.warning(f"{project_name}.{experiment_id} failed with NaNs")
        except BaseException as e:
            self.status = JobStatus.FAIL
            self.exp.logger.exception(f"{project_name}.{experiment_id} failed")
            if isinstance(e, KeyboardInterrupt):
                raise e
        else:
            self.status = JobStatus.DONE
            self.exp.logger.info(f"{project_name}.{experiment_id} done")

        if not keep:
            del self.exp

class ExperimentManager():
    side_runner = None
    def __init__(self, filename, logfile=None, num_workers=0, dummy=False, **grid_search):
        self.logger = logging.getLogger("experimentator")
        if logfile:
            handler = logging.FileHandler(logfile, mode="w")
            handler.setFormatter(logging.Formatter("[worker#%(threadName)s] %(asctime)s [%(levelname)s]%(filename)s:%(lineno)d: %(message)s"))
            handler.setLevel(logging.INFO if num_workers > 0 else logging.DEBUG)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if num_workers > 0 else logging.DEBUG)
        #threading.current_thread().name = "main"
        if num_workers > 0:
            self.side_runner = SideRunner(num_workers, impl=multiprocessing.Pool)#, impl=NestablePool)
            if "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == num_workers and num_workers > 1:
                self.side_runner.pool.map(set_cuda_visible_device, range(len(self.side_runner)))

        with open(find(filename)) as f:
            tree = ast.parse(f.read())

        if not grid_search:
            self.jobs = [Job(filename, config_tree=tree, dummy=dummy)]
        else:
            self.jobs = []
            unoverwritten = {}
            for grid_sample in product_kwargs(**grid_search):
                job = Job(filename, config_tree=copy.deepcopy(tree), dummy=dummy, grid_sample=grid_sample)
                unoverwritten.update(**job.update_ast(**grid_sample))
                self.jobs.append(job)
            if unoverwritten:
                self.logger.warning("Un-overwritten kwargs: {}".format(list(unoverwritten.keys())))

    @cached_property
    def worker_ids(self):
        if not self.side_runner:
            return {get_worker_id():0}
        seq = range(len(self.side_runner))
        return dict(zip(self.side_runner.pool.map(get_worker_id, seq), seq))

    def execute(self, epochs, **runtime_cfg):
        self.logger.info(f"Runtime config: {runtime_cfg}")
        for job in self.jobs:
            if job.status == JobStatus.TODO:
                if self.side_runner:
                    self.side_runner.run_async(Job.run, job, epochs=epochs, keep=False, worker_ids=self.worker_ids, **runtime_cfg)
                else:
                    #p = multiprocessing.Process(target=job.run, args=(epochs, False), kwargs=runtime_cfg)
                    #p.start()
                    #p.join()
                    job.run(epochs=epochs, keep=False, **runtime_cfg) # pylint: disable=expression-not-assigned

        if self.side_runner:
            self.side_runner.collect_runs()

def main():
    parser = argparse.ArgumentParser(description="Experimentation library", prog="experimentator")
    parser.add_argument("filename")
    parser.add_argument("--epochs", type=int)
    parser.add_argument('--logfile', type=str, default=None)# type=argparse.FileType('w', encoding='UTF-8')
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--grid', nargs="*")
    parser.add_argument('--kwargs', nargs="*", action='append')
    parser.add_argument('--dummy', default=False, action='store_true')
    args = parser.parse_args()

    # TODO: to be protected inside the if __name__ == '__main__' clause of the main module.
    #multiprocessing.set_start_method("spawn")

    grid = {}
    for arg in args.grid or []:
        exec(arg, None, grid) # pylint: disable=exec-used

    kwargs = {}
    for kwarg in [kwarg for kwargs in args.kwargs or [[]] for kwarg in kwargs]: # Flattened appended kwargs
        exec(kwarg, None, kwargs) # pylint: disable=exec-used

    num_subprocesses = 0 if args.workers <= 1 else args.workers
    manager = ExperimentManager(args.filename, logfile=args.logfile, num_workers=num_subprocesses, dummy=args.dummy, **grid)
    manager.execute(args.epochs, **kwargs)


    # def dump(self, filename="joblist.index"):
    #     # Write index
    #     #index_filename = os.path.join(self.folder, os.path.splitext(self.basename)[0] + self.datetime_suffix + ".index")
    #     with open(filename, "a+") as f:
    #         for job in self.jobs:
    #             f.write("[{}]\t{}\t{}\n".format(job.status, job.filename, str(job.grid_sample)))
    #     print(f"job list successfully written to {filename}")

    # @classmethod
    # def load(cls, filename="joblist.index"):
    #     worker_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    #     # Get job
    #     line = find_and_replace_wrapper(filename,
    #         lambda l: l.startswith("[TODO]"),
    #         lambda l: l.replace("[TODO]","[BUSY]").replace("\n","\t{}\n".format(worker_id))
    #     )

    #     if not line:
    #         print("nothing to do")
    #         return

    #     # Launch job
    #     try:
    #         _, filename, grid_sample = line.replace("\n","").split("\t")
    #         grid_sample = ast.literal_eval(grid_sample)
    #         run_job(filename, grid_sample=grid_sample, worker_id=worker_id, **kwargs)
    #     except:
    #         # Notify failure
    #         find_and_replace_wrapper(index_filename,
    #             lambda l: l.startswith("[BUSY]") and l.endswith(worker_id+"\n"),
    #             lambda l: l.replace("[BUSY]","[TODO]").replace("\t"+worker_id,"")
    #         )
    #         raise
    #     else:
    #         # Finish job
    #         line = find_and_replace_wrapper(index_filename,
    #             lambda l: l.startswith("[BUSY]") and l.endswith(worker_id+"\n"),
    #             lambda l: l.replace("[BUSY]","[DONE]")
    #         )

# def find_and_replace_wrapper(filename, search_expr, action_expr):
#     with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
#         output = None
#         for line in file:
#             if search_expr(line):
#                 print(action_expr(line), end="")
#                 output = line
#             else:
#                 print(line, end="")
#     return output

