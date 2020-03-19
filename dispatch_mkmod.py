#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model structure(s) with npz constraints - input constraints, output structure
"""

# builtin
import os
import json
import shlex
import shutil
import random
import secrets
import warnings
import argparse
import itertools
import subprocess
from enum import Enum
from time import sleep
from pathlib import Path
from datetime import datetime
from functools import partial
from importlib import import_module

# third-party
from dask.distributed import (Client, LocalCluster, as_completed,\
                              get_client, secede, rejoin)
from dask_jobqueue import SLURMCluster
import numpy as np

# local
from cst_toolbox import mkmod
from cst_toolbox.misc import WorkingDirectory, exists, nat


os.environ["OPENBLAS_NUM_THREADS"] = "1"

#########################
#>>>>> Commandline configuration 
#########################

def io_exists(arg, delim=' ', to_list=True):
    """Tuple of (exists, Path)"""
    i, o = arg.strip().split(delim)
    tup  = (exists(i), Path(o))
    return [tup] if to_list else tup 

def filelist(f):
    """File of io_exists 'types'""" 
    f = exists(f)

    tupler = partial(io_exists, to_list=False)
    with open(f, 'r') as fp:
        return list(map(tupler, fp))

class DefaultArguments(Enum):
    """Default command line parameters"""
    cluster            = True 
    overwrite          = False
    mode               = 0
    K                  = 50
    N                  = 150
    partition          = 'ccb'
    nodes              = 5

DEFAULT_PARAMS = {param.name: param.value for param in itertools.chain(DefaultArguments, mkmod.DefaultArguments)}

def arguments():
    """Configure commandline parser"""
    parser = argparse.ArgumentParser(description="Model structure with npz constraints using PyRosetta.")
    
    # I/O 
    parser.add_argument("--io-list", dest='i_o', type=filelist,
                          help="File-List of the form IN\sOUT\n",
                          metavar="FILE")

    parser.add_argument("-io", dest='i_o', help="InputNPZ OutputDirectory",
                          type=io_exists, metavar="IN OUT")
    
    # cluster arguments
    parser.add_argument("-N", "--num-workers", 
                        type=nat, dest='nodes', help="Number of parallel workers.") 
    parser.add_argument("-p", "--cluster-partition",
                        type=str, dest='partition',
                        choices=['ccb', 'bnl'], help="SLURM partition")
    parser.add_argument("--local", help="Local multithreaded mode", 
                        action='store_false', dest='cluster')

    # input parameters
    parser.add_argument("-n","--nstruct", help="Number of coarse-grained structures to generate",
                        type=nat, dest='N')
    parser.add_argument("-k","--nrelax", help="Number (k < n) of structures to perform full-atom refinement", type=nat,
                        dest='K')

    # mkmod parameters
    mkmod._setup_arguments(parser, defaults=False)
    
    parser.set_defaults(**DEFAULT_PARAMS)

    return parser.parse_args()


def compute_top_k(models, k, tsvfile):
    """Compute top k models and record the scores of all of them"""
    topk = []

    with open(tsvfile, 'w') as tsv:
        tsv.write("path\tscore\n")
        for i, x in enumerate(sorted(models, key=lambda x:x['score'])):
            path, score = x['path'], x['score']
            tsv.write(f"{path}\t{score}\n")
            if i < k:
                topk.append(x)

    return topk 

#########################
#>>>>> Helper functions to manage distributed computing
#########################

class Factory(object):
    def __init__(self, Class):
        self.blueprint = Class

    def build(self, *args, **kwargs):
        print(f"Building {self.blueprint.__name__}")
        return self.blueprint(*args, **kwargs)

def get_cluster(partition=None, distributed=False):
    params = {}
    if distributed:
        params['processes'] = 28
        params['cores']     = 1 
        params['memory']    = '224GB'
        params['queue'] = partition
        clust = SLURMCluster
    else:
        clust = LocalCluster
    return Factory(clust).build(**params)

class Manager(object):
    """
    Manages work and futures, blocks until finished once
    flagged to do so.
    """
    def __init__(self, func=None, todo=[]):
        self.__todo = todo  # stack of todo items
        self.__func = func
        self.__running = []
        self.__results = []

    @property
    def func(self):
        return self.__func
    
    def add_work(self, item, submit=True):
        self._amend_todo(item)
        if submit:
            self.submit()

    def _amend_todo(self, item):
        self.__todo.append(item)

    @property
    def primed(self):
        return len(self.__todo) != 0

    def submit(self):
        if self.__func is not None:
            if len(self.__todo):
                client = get_client()
                item = self.__todo.pop() 
                fut = client.submit(self.__func, *item)
                self.__running.append(fut)
            return len(self.__running) 
        else:
            raise RuntimeError("No func enabled")

    def results(self):
        """Blocks until complete"""
        # submit remaining jobs
        while self.primed:
            self.submit()
        for fut in as_completed(self.__running):
            self.__results.append(fut.result())

        return self.__results

    def as_completed(self, drain=True):
        """Emit submitted jobs as completed,
        drain all from the work queue if specified"""
        if drain:
            while self.primed:
                self.submit()
        yield from map(lambda fut: fut.result(), as_completed(self.__running))

def run_mkmod_command(cmd, outjson):
    """Runs mkmod assuming the command contained an --output_json parameter
    and retrives that json information
    
    args:
        :cmd (str) - command string
        :outjson (str) - path to expected output file
    returns:
        :(dict) - json structure
    """
    command = shlex.split(cmd)

    # blocks until complete
    subprocess.run(command) 
    with open(outjson, 'r') as js:
        jstruct = json.loads(js.read())
    return jstruct
    
def make_models(input_npz, output_dir, params):
    """Run the entire modeling workflow, which basically consists of dispatching mkmod calls
    args:
        :input_npz (str or Path) - path to input npz file
        :output_dir (str or Path) - path to output directory
        :params (dict) - Command line parameters (see arguments())
    returns:
        :(str) - path to final model
    """
    print(input_npz, output_dir)
    #########
    start = datetime.now()
    #########

    tmpdir = WorkingDirectory(path=output_dir, cleanup=False).setup()
    input_npz = str(input_npz)

    params['TDIR'] = output_dir
    
    # create a models directory
    models = tmpdir.dirname / 'models'
    tmp    = tmpdir.dirname / 'tmp'
    models.mkdir(exist_ok=True, parents=True)
    tmp.mkdir(exist_ok=True, parents=True)
    params['models'] = str(models)
    params['tmp']    = str(tmp)

    # submit centroid model work
    centroid_mgr = Manager(func=run_mkmod_command)
    
    configurables  = "--model_id {model_id} --model_json {model_json} " 
    command        = f"python {mkmod.PATH} {input_npz} {output_dir} "
    
    csts = " ".join(params['constraint_types'])
    flags  = f"--constraint-types {csts} "
    #flags += f"--score-fxn-wts {params['score_function_dir']}"
    if params['overwrite']:
        flags += "--overwrite"

    command_format = command + configurables + flags 

    for i in range(params['N']):
        mid = f"{i+1:03d}_of_{params['N']}"
        outjson = str(tmp / f"{mid}_{secrets.token_hex(16)}.json")
        cmd = command_format.format(model_id=mid, model_json=outjson)
        centroid_mgr.add_work((cmd, outjson), submit=True)

    centroids = centroid_mgr.results() 
    # adjust the atom dist maximum for restraint setup
    # so that relaxation is more centralized around locality
    #params['ATOM_DIST_MAX'] = 10.0
    
    # record top k centroids
    topk = compute_top_k(centroids, params['K'], str(tmpdir.dirname / 'centroid-models.tsv'))
    print(f"[{output_dir}] Got {len(topk)} top scoring centroid models.")
    
    # add a relax parameter ot config flgs.
    configurables += "--relax {input_pdb} "
    command_format = command + configurables + flags

    relax_mgr  = Manager(func=run_mkmod_command)
    for t in topk:
        mid = t['model_id']
        outjson = str(tmp / f"{mid}_relax_{secrets.token_hex(16)}.json")
        input_pdb = t['path']
        cmd = command_format.format(model_id=mid, model_json=outjson, input_pdb=input_pdb)
        relax_mgr.add_work((cmd, outjson), submit=True)

    relaxed = relax_mgr.results()
    top1 = compute_top_k(relaxed, 1, str(tmpdir.dirname / 'relaxed-models.tsv'))
    shutil.copyfile(top1[0]['path'], tmpdir.dirname / 'final.pdb')
    #########
    finish = datetime.now()
    #########
    print(f"{tmpdir.dirname.name}: {finish - start}")
    return str(tmpdir.dirname / 'final.pdb')

if __name__ == '__main__':
    args = arguments()
    if args.K > args.N:
        args.K = args.N
        warnings.warn(f"Relax models ({args.K}) > Centroid models ({args.N}). Setting them equal.",
                      UserWarning)
    
    args.one = len(args.i_o) == 1
    
    params = DEFAULT_PARAMS
    params.update(vars(args).items())
    del params['i_o']
    os.environ['PYROSETTA_BOOTED'] = "yes"

    cluster = get_cluster(distributed=args.cluster, partition=args.partition)
    client  = Client(cluster)
    print("Created cluster")
    cluster.scale(args.nodes)
    print("Called scale")
    
    big_mgr = Manager(make_models)
    print("Dispatching jobs")
    finals = []
    try:
        for i, (input_npz, output_dir) in enumerate(args.i_o):
            if not (Path(output_dir) / 'final.pdb').exists():
                start = datetime.now()
                big_mgr.add_work((input_npz, output_dir, params), submit=True)
            else:
                print(f"{input_npz} is finished!")
    
        finals = big_mgr.results()
    finally:
        print("Done!")
        client.shutdown()
