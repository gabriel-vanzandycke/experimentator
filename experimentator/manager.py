import argparse
import datetime
from enum import Enum
from functools import cached_property
import itertools
import logging
import multiprocessing
import os
import time

from mlworkflow import SideRunner
from experimentator import DummyExperiment
from experimentator.utils import find, mkdir
from experimentator.callbacked_experiment import FailedTrainingError
from experimentator.base_experiment import BaseExperiment
from pyconfyg import GridConfyg, parse_strings, Confyg
# pylint: disable=logging-fstring-interpolation, logging-format-interpolation

def parse_config_file(config_filename):
    with open(config_filename, "r") as f:
        config = parse_strings(f.read())
    return config

def get_worker_id(*_):
    time.sleep(.1)
    return os.getpid()

def set_cuda_visible_device(index):
    time.sleep(.1)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(',')[index]

def build_experiment(config_filename, **kwargs) -> BaseExperiment:
    confyg = Confyg(config_filename, dict({**kwargs, "filename": config_filename}))
    exp = type("Exp", tuple([t for t in confyg.dict["experiment_type"][::-1] if t is not None]), {})(confyg.dict)
    exp.load_weights()
    return exp

class JobStatus(Enum):
    TODO = 0
    BUSY = 1
    FAIL = 2
    DONE = 3

class Job():
    def __init__(self, confyg: Confyg, project_name: str, dummy=False, grid_sample=None):
        self.dummy = dummy
        self.confyg = confyg
        self.grid_sample = grid_sample or {}
        self.status = JobStatus.TODO
        self.project_name = project_name

    @cached_property
    def exp(self):
        if self.dummy:
            self.confyg.dict["experiment_type"].append(DummyExperiment)
        Type = type("Exp", tuple([t for t in self.confyg.dict["experiment_type"][::-1] if t is not None]), {})
        return Type(self.confyg.dict)

    def run(self, epochs, keep=True, worker_ids=None):
        experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        worker_index = worker_ids[get_worker_id()] if worker_ids else 0

        # Write config string to file
        folder = os.path.join(os.getenv("RESULTS_FOLDER", "."), self.project_name, experiment_id)
        mkdir(folder)
        link = os.path.join(os.getenv("RESULTS_FOLDER", "."), self.project_name, "latest")
        os.remove(link)
        os.symlink(folder, link)
        filename = os.path.join(folder, "config.py")
        with open(filename, "w") as f:
            f.write(self.confyg.string)

        # Add run and project names
        self.confyg.dict.update(
            project_name=self.project_name,
            experiment_id=experiment_id,
            worker_index=worker_index,
            folder=folder,
            dummy=self.dummy,
            grid_sample=self.grid_sample
        )

        # Launch training
        try:
            self.status = JobStatus.BUSY
            self.exp.logger.info(f"{self.project_name}.{experiment_id} doing {self.grid_sample}")
            self.exp.train(epochs=epochs)
        except FailedTrainingError as e:
            self.status = JobStatus.FAIL
            self.exp.logger.warning(f"{self.project_name}.{experiment_id} failed with NaNs")
        except BaseException as e:
            self.status = JobStatus.FAIL
            self.exp.logger.exception(f"{self.project_name}.{experiment_id} failed")
            if isinstance(e, KeyboardInterrupt):
                raise e
        else:
            self.status = JobStatus.DONE
            self.exp.logger.info(f"{self.project_name}.{experiment_id} done")

        if not keep:
            del self.exp

class ExperimentManager():
    side_runner = None
    def __init__(self, filename, logfile=None, num_workers=0, dummy=False, grid_search=None, runtime_cfg=None):
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

        self.logger.info(f"Runtime config: {runtime_cfg}")
        gc = GridConfyg(find(filename), grid_search, runtime_cfg)
        project_name = runtime_cfg.get("project_name", os.path.splitext(os.path.basename(filename))[0])
        self.jobs = [Job(confyg, dummy=dummy, project_name=project_name, grid_sample=grid_sample) for grid_sample, confyg in iter(gc)]

    @cached_property
    def worker_ids(self):
        if not self.side_runner:
            return {get_worker_id():0}
        seq = range(len(self.side_runner))
        return dict(zip(self.side_runner.pool.map(get_worker_id, seq), seq))

    def execute(self, epochs):
        for job in self.jobs:
            if job.status == JobStatus.TODO:
                if self.side_runner:
                    self.side_runner.run_async(Job.run, job, epochs=epochs, keep=False, worker_ids=self.worker_ids)
                else:
                    #p = multiprocessing.Process(target=job.run, args=(epochs, False), kwargs=runtime_cfg)
                    #p.start()
                    #p.join()
                    job.run(epochs=epochs, keep=False) # pylint: disable=expression-not-assigned

        if self.side_runner:
            self.side_runner.collect_runs()

def main():
    parser = argparse.ArgumentParser(description="Experimentation library", prog="experimentator")
    parser.add_argument("filename")
    parser.add_argument("--epochs", type=int)
    parser.add_argument('--logfile', type=str, default=None)# type=argparse.FileType('w', encoding='UTF-8')
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--grid', nargs="*", action='append', default=[[]])
    parser.add_argument('--kwargs', nargs="*", action='append', default=[[]])
    parser.add_argument('--dummy', default=False, action='store_true')
    args = parser.parse_args()

    # TODO: to be protected inside the if __name__ == '__main__' clause of the main module.
    #multiprocessing.set_start_method("spawn")

    grid = parse_strings(*list(itertools.chain(*args.grid)))
    kwargs = parse_strings(*list(itertools.chain(*args.kwargs)))

    num_subprocesses = 0 if args.workers <= 1 else args.workers
    manager = ExperimentManager(args.filename, logfile=args.logfile, num_workers=num_subprocesses, dummy=args.dummy, grid_search=grid, runtime_cfg=kwargs)
    manager.execute(args.epochs)
