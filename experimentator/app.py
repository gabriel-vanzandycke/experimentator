#!/usr/bin/env python3

import argparse
import os
from .utils import find
from .manager import ExperimentManager

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=-1, help="Number of epochs to train (if not provided or 0, will prompt)")
    parser.add_argument("--config", required=True, help="Experiment '.ini' configuration file. If no config is provided, prompts to loads a previous process")
    parser.add_argument("--gpu", default=0, help="Default GPU to use")
    parser.add_argument("--cpu", action="store_true", help="Default GPU to use")
    parser.add_argument("--robust", action="store_true", help="Should experiments be encapsulated inside a try-except")
    parser.add_argument("--name", default=None, help="Config file name")
    parser.add_argument("--eager", action="store_true", help="should be run in eagermode")
    parser.add_argument('--args', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    #assert args.gpu is not None or args.cpu, "You must specify either cpu or gpu"

    try:
        import provision#from provision import *
    except BaseException as e:
        raise e
    
    kwargs = {}
    if args.args:
        for arg in args.args:
            assert "=" in arg, "= not in arg: ".format(arg)
            name, value = arg.split("=")
            kwargs[name] = value
    print("Additional arguments: ", kwargs)

    manager = ExperimentManager(find(args.config), robust=args.robust, name=args.name, gpu=args.gpu, eager=args.eager, **kwargs)
    if not args.config:
        manager.load()
    if args.epochs < 0:
        args.epochs = int(input("How many epochs ?"))

    manager.execute(args.epochs)
