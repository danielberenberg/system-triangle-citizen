#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model structure(s) with npz constraints - input constraints, output structure

"""
#TODO(make this work for multiple npz inputs)

# builtin
import os
import heapq
import shlex
import shutil
import warnings
import argparse
from enum import Enum
from pathlib import Path
from functools import partial
from importlib import import_module

# third-party
import cloudpickle
import joblib
from dask.distributed import Client, LocalCluster, as_completed, get_client
from dask_jobqueue import SLURMCluster

import numpy as np

# local

from toolbox import rosetta_utils
from toolbox.misc import WorkingDirectory

_DEFAULT_SCORE_FILES = Path(__file__).parent / 'data' / 'sfxn'
if not _DEFAULT_SCORE_FILES.is_dir():
    _DEFAULT_SCORE_FILES = None

ROSETTA_LOGLEVEL = 100

os.environ["OPENBLAS_NUM_THREADS"] = "1"

#########################
#>>>>> Commandline configuration 
#########################

def nat(x):
    """Natural number > 0 'type'"""
    x = int(x)
    if x <= 0:
        raise TypeError(f"Expected natural number, not {x}")
    return x

def exists(f):
    """Existing path 'type'"""
    f = Path(f)
    if not f.exists():
        raise FileNotFoundError(f"{f} doesn't exist")
    return f 

def io_exists(arg, delim=' ', to_list=True):
    """Tuple of (exists, Path)"""
    i, o = arg.split(delim)
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
    constraint_types   = ['dist', 'theta', 'omega', 'phi']
    score_function_dir = _DEFAULT_SCORE_FILES 
    cluster            = True 
    mode               = 0
    K                  = 50
    N                  = 150
    partition          = 'bnl'
    nodes              = 5

DEFAULT_PARAMS = {param.name: param.value for param in DefaultArguments}


def arguments():
    """Configure commandline parser"""
    parser = argparse.ArgumentParser(description="Model structure with npz constraints using PyRosetta.")
    
    # input possibilities
    parser.add_argument("--io-list", dest='i_o', type=filelist,
                          help="Two column (input npz/output dir) filelist")

    parser.add_argument("-io", dest='i_o', help="InputNPZ OutputDirectory",
                          type=io_exists)
    
    # cluster arguments
    parser.add_argument("-N", "--cluster-nodes", type=nat, dest='nodes', help="Number of cluster nodes to scale") 
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

    parser.add_argument("-C", "--constraint-types", nargs='+',
                        choices=[cst.name for cst in rosetta_utils.ConstraintTypes],
                        dest='constraint_types', 
                        help="Include only these input channels from the NPZ.")

    parser.add_argument("-M", "--mode", choices=[0, 1, 2], dest='mode',
                        type=int, help="Run mode: 0=short->medium->long, 1=short+medium->long, 2=short+medium+long")
    
    # Rosetta parameters
    parser.add_argument("--rosetta-init", type=str,
                        dest='initargs',
                        help="Flags to pass to pyrosetta.init (wrap this in doub. quotes).",
                        default=f"-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level {ROSETTA_LOGLEVEL}")

    parser.add_argument("--score-fxn-wts", dest='score_function_dir',
                        help=f"Score function weights directory. (Default={DefaultArguments.score_function_dir.value or 'Not found...'})",
                        required=DefaultArguments.score_function_dir.value is None)

    parser.set_defaults(**DEFAULT_PARAMS)

    return parser.parse_args()

#########################
#>>>>> Rosetta functions 
#########################
def minimize(seq, rst, params):
    """
    Worker fully encapsulating the minimization protocol.

    Generates a random pose using the input sequence, minimizes with
    respect to the restraints given by `rst` and the specified mode, dumps the pose into 
    the tempdir found in `params`
    args:
        :seq (str) - sequence
        :rst (dict-like) - Generated constraints
    returns:
        :(dict): {'path': str, 'score': score}
    """
    pyrosetta = import_module("pyrosetta")
    pyrosetta.distributed.init(params['initargs'])
    os.environ['PYROSETTA_BOOTED'] = "yes"

    model_id = params.get("model_id") or secrets.token_hex(16)
    print(f"[{model_id}] BEGIN MINIMIZE")

    #####=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-##### 
    #| setup ScoreFunctions and Mover objects |#
    #####=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-##### 
    sfdir = Path(params['score_function_dir'])
    sfs   = [str(sfdir / sffn) for sffn in ['scorefxn.wts', 'scorefxn1.wts', 'scorefxn_vdw.wts', 'scorefxn_cart.wts']]
    sf, sf1, sf_vdw, sf_cart = map(rosetta_utils.load_score_fxn_with_weights, sfs)     
   
    def _setup_movemap(bb=True, chi=False, jump=True):
        mmap = pyrosetta.MoveMap()
        mmap.set_bb(bb)
        mmap.set_chi(chi)
        mmap.set_jump(jump)
        return mmap

    mmap = _setup_movemap()

    def _setup_minmover(fxn, 
                        opt='lbfgs_armijo_nonmonotone',
                        cutoff=0.0001,
                        max_iter=1000, cartesian=None):
        MinMover = pyrosetta.rosetta.protocols.minimization_packing.MinMover
        # movemap, score function, optimizer, convegence cutoff
        mover = MinMover(mmap, fxn, opt, cutoff, True)
        mover.max_iter(max_iter)
        if cartesian is not None:
            mover.cartesian(cartesian)
        return mover

    min_mover      = _setup_minmover(sf, max_iter=1000) 
    min_mover1     = _setup_minmover(sf1, max_iter=1000) 
    min_mover_vdw  = _setup_minmover(sf_vdw, max_iter=500)
    min_mover_cart = _setup_minmover(sf_cart, cartesian=True, max_iter=1000)
    repeat_mover   = pyrosetta.RepeatMover(min_mover, 3)

    ## setup pose
    pose =  rosetta_utils.load_pose(seq, from_sequence=True)
    rosetta_utils.set_random_dihedral(pose)
    rosetta_utils.remove_clash(sf_vdw, min_mover_vdw, pose)

    if params['mode'] == 0:
        schedule = [1, 12, 24, len(seq)]
    elif params['mode'] == 1:
        schedule = [1, 24, len(seq)]
    elif params['mode'] == 2:
        schedule = [1, len(seq)]

    # mutate GLY -> ALA so every AA has a CB
    with rosetta_utils.ContextMutator("G", "A", pose) as cm:
        for interval in zip(schedule[:-1], schedule[1:]):
            print(f"[{model_id}] begin range {interval}")
            rosetta_utils.add_constraints(pose, rst, interval, seq, params['TDIR'])
            repeat_mover.apply(pose)
            min_mover_cart.apply(pose)
            rosetta_utils.remove_clash(sf_vdw, min_mover1, pose)
            print(f"[{model_id}] done range {interval}")
    
    centroid_score_function = pyrosetta.create_score_function('cen_std')
    score = centroid_score_function(pose)

    print(f"[{model_id}] Centroid score: {score}")
    dumpfile = str(os.path.join(params['models'], "model_" + model_id + '.pdb')) 
    pose.dump_pdb(dumpfile)

    print(f"[{model_id}] END MINIMIZE")
    return {'path': dumpfile, 'score': score, 'model_id': model_id}


def relax(filename, rst, params):
    """Encapsulates the relax protocol
    
    args:
        :rst (dict-like) - Generated constraints
    returns:
        :(dict): {'path': str, 'score': score}
    """
    pyrosetta = import_module("pyrosetta")
    pyrosetta.distributed.init(params['initargs'])
    os.environ['PYROSETTA_BOOTED'] = "yes"

    model_id = params.get("model_id") or secrets.token_hex(16)
    print(f"[{model_id}] BEGIN RELAX")

    def _setup_movemap(bb=True, chi=False, jump=True):
        mmap = pyrosetta.MoveMap()
        mmap.set_bb(bb)
        mmap.set_chi(chi)
        mmap.set_jump(jump)
        return mmap

    pose = rosetta_utils.load_pose(filename)

    sf_fa = pyrosetta.create_score_function('ref2015')
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 5)
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1)
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1)
        
    mmap = _setup_movemap(bb=True, chi=True, jump=True)
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(200)
    relax.dualspace(True)
    relax.set_movemap(mmap)

    pose.remove_constraints()
    
    # switch to full atom mode
    switch = pyrosetta.SwitchResidueTypeSetMover("fa_standard")
    switch.apply(pose)
    
    seq = pose.sequence()
    rosetta_utils.add_constraints(pose, rst, (1, len(seq)), seq, params['TDIR'], nogly=True)
    relax.apply(pose)

    score = sf_fa(pose)
    print(f"[{model_id}] FastRelax fa_standard score: {score}")
    dumpfile = str(os.path.join(params['models'], 'relax_' + model_id + '.pdb'))
    pose.dump_pdb(dumpfile)
    print(f"[{model_id}] END RELAX")

    return {'path': dumpfile, 'score': score, 'model_id': model_id}


def compute_top_k(models, k, tsvfile):
    """Compute top k models and record the scores of all of them"""
    topk = []

    def key(x):
        if isinstance(x, tuple):
            return x[0]
        elif isinstance(x, dict):
            return x['score']

    def vals(x):
        if isinstance(x, tuple):
            return x
        elif isinstance(x, dict):
            return x['path'], x['score']

    with open(tsvfile, 'w') as tsv:
        tsv.write("path\tscore\n")
        for i, (path, score) in enumerate(map(vals, sorted(models, key=key))):
            tsv.write(f"{path}\t{score}\n")
            if i < k:
                topk.append({'path': path, 'score': score})
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
        params['cores'] = 20
        params['memory'] = '25GB'
        params['queue'] = partition
        clust = SLURMCluster
    else:
        clust = LocalCluster
    print("Getting a {clust} built") 
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

def make_models(input_npz, output_dir, params):
    """Run the entire modeling workflow
    args:
        :input_npz (str or Path) - path to input npz file
        :output_dir (str or Path) - path to output directory
        :params (dict) - Command line parameters (see arguments())
    returns:
        :(str) - path to final model
    """
    tmpdir = WorkingDirectory(path=output_dir, cleanup=False).setup()
    npz = np.load(input_npz, allow_pickle=True)
    seq = "".join(np.atleast_1d(npz['sequence']))

    params['TDIR'] = output_dir
    
    # create a models directory
    models = tmpdir.dirname / 'models'
    models.mkdir(exist_ok=True, parents=True)
    params['models'] = str(models)
    
    # create first round of restraints
    rst = rosetta_utils.generate_constraints(npz, **params)

    # submit centroid model work
    centroid_mgr = Manager(func=minimize)
    relax_mgr    = Manager(func=relax)
    for i in range(params['N']):
        mid = f"{i+1:03d}_of_{params['N']}"
        centroid_mgr.add_work((seq, rst, {**params, 'model_id': mid}), submit=False)
    
    # relax step
    params['ATOM_DIST_MAX'] = 10.0
    # adjust the atom dist maximum for restraint setup
    # so that relaxation is more centralized around locality
    print("Regenerating restraints")
    rlxrst = rosetta_utils.generate_constraints(npz, **params)

    # Relax models as they come in 
    centroids = [] 
    submitted = 0
    heapq.heapify(centroids)
    max_top_score = -np.inf
    for centroid in centroid_mgr.as_completed():
        heapq.heappush(centroids, (centroid['score'], centroid['path']))
        # start K rleax models right on the outset and then be conservative 
        # by only submitting relax jobs to those that are within the min K scores 
        if submitted < params['K'] or centroid['score'] <= max_top_score: # start K relax modes right on the outset
            relax_mgr.add_work((centroid['path'], rlxrst, {**params, 'model_id': centroid['model_id']}), submit=True)
            max_top_score = max(heapq.nsmallest(params['K'], centroids), key=lambda x: x[0])[0]
            submitted += 1
        else:
            print(f"Skipping relax run for {centroid['path']} ({centroid['score']} > {max_top_score})")

    # record top k centroids
    topk = compute_top_k(centroids, params['K'], str(tmpdir.dirname / 'centroid-models.tsv'))
    relaxed = relax_mgr.results()
    top1 = compute_top_k(relaxed, 1, str(tmpdir.dirname / 'relaxed-models.tsv'))
    return top1[0]


if __name__ == '__main__':
    args = arguments()
    print(args)
    if args.K > args.N:
        args.K = args.N
        warnings.warn(f"Relax models ({args.K}) > Centroid models ({args.N}). Setting them equal.",
                      UserWarning)
    
    params = DEFAULT_PARAMS
    params.update(vars(args).items())

    cluster = get_cluster(distributed=args.cluster, partition=args.partition)
    client  = Client(cluster)
    print("Created cluster")
    cluster.scale(args.nodes)
    print("Called scale")

    big_mgr = Manager(make_models)
    for input_npz, output_dir in args.i_o:
        big_mgr.add_work((input_npz, output_dir, params), submit=True)

    finals = big_mgr.results()

    for path in map(lambda x: Path(x['path']), finals):
        print(path)
        shutil.copyfile(path, path.parent.parent / 'final.pdb')
    
    print("Done!")
    client.shutdown()
