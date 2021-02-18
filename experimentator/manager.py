import argparse
import ast
import datetime
from enum import Enum
#import fileinput
import itertools
import os
import astunparse
from mlworkflow import SideRunner, lazyproperty
from .utils import find

def product_kwargs(**kwargs):
    try:
        kvs = [[(k, v) for v in kwargs[k]] for k in kwargs]
    except:
        print(f"Error parsing: {kwargs}")
        raise
    yield from [dict(kv) for kv in itertools.product(*kvs)]

def update_ast_tree(tree, overwrite):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                # Deal with tuple assignments (a,b = 1,2)
                continue
            if target.id in overwrite:
                assert isinstance(node.value, ast.Constant), "Only overwritting constants is currently supported"
                # Replace node value by the value in overwrite
                node.value.value = overwrite.pop(target.id)
    # Add remaining keys
    for key, value in overwrite.items():
        tree.body.append(ast.Assign([ast.Name(id=key, ctx=ast.Store())], ast.Constant(value, kind=None)))
    ast.fix_missing_locations(tree)
    return tree

def insert_suffix(filename, suffix):
    root, ext = os.path.splitext(filename)
    return root + suffix + ext

class JobStatus(Enum):
    TODO = 0
    BUSY = 1
    FAIL = 2
    DONE = 3

class Job():
    def __init__(self, filename, config, grid_sample=None):
        self.filename = filename
        self.config = config
        self.grid_sample = grid_sample or {}
        self.status = JobStatus.TODO

    @lazyproperty
    def exp(self):
        os.sys.path.insert(0, os.getcwd())
        config = {}
        exec(self.config, None, config) # pylint: disable=exec-used
        config = {**config, "grid_sample": self.grid_sample, "config": self.config, "filename": self.filename}
        return type("Exp", tuple(config["experiment_type"][::-1]), {})(config)

    def run(self, **kwargs):
        try:
            self.status = JobStatus.BUSY
            self.exp.update(**kwargs)
            self.exp.cfg["experiment_id"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
            self.exp.cfg["project_name"] = os.path.splitext(os.path.basename(self.exp.cfg["filename"]))[0]
            self.exp.train(epochs=kwargs["epochs"])
        except:
            self.status = JobStatus.FAIL
            raise
        else:
            self.status = JobStatus.DONE

class ExperimentManager():
    def __init__(self, filename, num_workers=0, **grid_search):
        print("config: {}\tgrid_search: {}".format(filename, grid_search), flush=True)
        #self.filename = filename                                                  # unused
        #self.basename = os.path.basename(self.filename)                           # unused
        #self.datetime_suffix = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S") # unused
        self.side_runner = SideRunner(thread_count=num_workers) if num_workers > 0 else None

        with open(find(filename)) as f:
            if not grid_search:
                self.jobs = [Job(filename, f.read())]
            else:
                self.jobs = []
                tree = ast.parse(f.read())
                for grid_sample in product_kwargs(**grid_search):
                    update_ast_tree(tree, dict(grid_sample)) # dict makes a copy
                    self.jobs.append(Job(filename, config=astunparse.unparse(tree), grid_sample=grid_sample))

    def execute(self, epochs, **kwargs):
        print("overridden config {}".format(kwargs))
        for job in self.jobs:
            if job.status == JobStatus.TODO:
                print("Doing {}".format(job.grid_sample), flush=True)
                self.side_runner.do(Job.run, job, epochs=epochs, **kwargs) if self.side_runner else job.run(epochs=epochs, **kwargs) # pylint: disable=expression-not-assigned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--epochs", type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--grid', nargs="*")
    parser.add_argument('--kwargs', nargs="*")
    args = parser.parse_args()

    grid = {}
    for arg in args.grid or []:
        exec(arg, None, grid) # pylint: disable=exec-used

    kwargs = {}
    for arg in args.kwargs or []:
        exec(arg, None, kwargs) # pylint: disable=exec-used

    manager = ExperimentManager(args.filename, num_workers=args.workers, **grid)

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

