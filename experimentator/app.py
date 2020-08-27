#!/usr/bin/env python3

import argparse
import os
from .utils import find
from .manager import ExperimentManager

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=-1, help="Number of epochs to train (if not provided or 0, will prompt)")
    parser.add_argument("--config", required=True, help="Experiment '.ini' configuration file. If no config is provided, prompts to loads a previous process")
    parser.add_argument("--gpu", required=True, help="Default GPU to use")
    parser.add_argument("--robust", action="store_true", help="Should experiments be encapsulated inside a try-except")
    parser.add_argument("--name", default=None, help="Config file name")

    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    try:
        import provision#from provision import *
    except BaseException as e:
        raise e

    #from provision import *  # pylint: disable=wildcard-import, unused-wildcard-import, wrong-import-position

    manager = ExperimentManager(find(args.config), robust=args.robust, name=args.name, gpu=args.gpu)
    if not args.config:
        manager.load()
    if args.epochs < 0:
        args.epochs = int(input("How many epochs ?"))

    manager.execute(args.epochs)
