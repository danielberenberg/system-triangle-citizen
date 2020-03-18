#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model structure(s) with npz constraints - input constraints, output structure

"""
#TODO(make this work for multiple npz inputs)

# builtin
import os
import shutil
import random
import secrets
import warnings
import argparse
import random
from enum import Enum
from time import sleep
from pathlib import Path
from datetime import datetime
from functools import partial
from importlib import import_module

# third-party
import cloudpickle  # not used here but needed for internal pickling magic
import joblib
from dask.distributed import (Client, LocalCluster, as_completed,\
                              get_client, secede, rejoin)
from dask_jobqueue import SLURMCluster

import numpy as np

# local
from cst_toolbox import rosetta_utils
from cst_toolbox.misc import WorkingDirectory, exists, nat


_DEFAULT_SCORE_FILES = Path(__file__).parent / 'data' / 'sfxn'
if not _DEFAULT_SCORE_FILES.is_dir():
    _DEFAULT_SCORE_FILES = None

ROSETTA_LOGLEVEL = 100

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
    constraint_types   = ['dist', 'theta', 'omega', 'phi']
    score_function_dir = _DEFAULT_SCORE_FILES 
    cluster            = True 
    overwrite          = False
    mode               = 0
    K                  = 50
    N                  = 150
    partition          = 'ccb'
    nodes              = 5
    initargs           =  f"-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level {ROSETTA_LOGLEVEL}"

DEFAULT_PARAMS = {param.name: param.value for param in DefaultArguments}


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

    parser.add_argument("-C", "--constraint-types", nargs='+',
                        choices=[cst.name for cst in rosetta_utils.ConstraintTypes],
                        dest='constraint_types', 
                        help="Include only these input channels from the NPZ.")

    parser.add_argument("-M", "--mode", choices=[0, 1, 2], dest='mode',
                        type=int, help="Run mode: 0=short->medium->long, 1=short+medium->long, 2=short+medium+long")
    
    # Rosetta parameters
    parser.add_argument("--rosetta-init", type=str,
                        dest='initargs',
                        help="Flags to pass to pyrosetta.init (wrap this in doub. quotes).")

    parser.add_argument("--score-fxn-wts", dest='score_function_dir',
                        help=f"Score function weights directory. (Default={DefaultArguments.score_function_dir.value or 'Not found...'})",
                        required=DefaultArguments.score_function_dir.value is None)

    parser.set_defaults(**DEFAULT_PARAMS)

    return parser.parse_args()

#########################
#>>>>> Rosetta functions 
#########################
def minimize(seq, input_npz, params):
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
    try:
        print(seq)
        pyrosetta = import_module("pyrosetta")
        #if params['init_pyrosetta']:
        pyrosetta.distributed.init(params['initargs'])

        npz = np.load(input_npz, allow_pickle=True)
        rst = rosetta_utils.generate_constraints(npz, **params)

        model_id = params.get("model_id") or secrets.token_hex(16)
        dumpfile = str(os.path.join(params['models'], "model_" + model_id + '.pdb')) 

        prefix = f"[{Path(params['TDIR']).name}-{model_id}]"
        try:
            centroid_score_function = pyrosetta.create_score_function('cen_std')
        except:
            snooze = random.randint(1, 10)
            sleep(snooze)
            centroid_score_function = pyrosetta.create_score_function('cen_std')

        overwrite = params['overwrite']
        if Path(dumpfile).exists() and not overwrite:
            pose = rosetta_utils.load_pose(dumpfile)
            switch = pyrosetta.SwitchResidueTypeSetMover("centroid")
            switch.apply(pose)
            score = centroid_score_function(pose)
            print(f"{prefix} (precomputed) Centroid score: {score}")
            #print(f"{prefix} END MINIMIZE")
            return {'path': dumpfile, 'score': score, 'model_id': model_id}

        #print(dumpfile, "does not exist")
        #if params['cluster']:
        print(f"{prefix} (minimize) seceding")
        secede()

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
                rosetta_utils.add_constraints(pose, rst, interval, seq, params['TDIR'])
                repeat_mover.apply(pose)
                min_mover_cart.apply(pose)
                rosetta_utils.remove_clash(sf_vdw, min_mover1, pose)

        score = centroid_score_function(pose)
        print(f"{prefix} (minimize) Centroid score: {score}")
        pose.dump_pdb(dumpfile)

        #print(f"{prefix} END MINIMIZE")
        #if params['cluster']:
        print(f"{prefix} (minimize) rejoining")
        rejoin()
        return {'path': dumpfile, 'score': score, 'model_id': model_id}
    except Exception as e:
        with open("except-minimize.txt", 'a') as f:
            print(f"args={(seq, input_npz, params)}\nexception={e}", file=f)


def relax(filename, input_npz, params):
    """Encapsulates the relax protocol
    
    args:
        :rst (dict-like) - Generated constraints
    returns:
        :(dict): {'path': str, 'score': score}
    """
    
    try:
        pyrosetta = import_module("pyrosetta")
        #if params['init_pyrosetta']:
        pyrosetta.distributed.init(params['initargs'])
        
        npz = np.load(input_npz, allow_pickle=True)
        rst = rosetta_utils.generate_constraints(npz, **params)

        model_id = params.get("model_id") or secrets.token_hex(16)
        prefix = f"[{Path(params['TDIR']).name}-{model_id}]"

        sf_fa = pyrosetta.create_score_function('ref2015')
        sf_fa.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, 5)
        sf_fa.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, 1)
        sf_fa.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, 1)
            
        dumpfile = str(os.path.join(params['models'], 'relax_' + model_id + '.pdb'))
        if Path(dumpfile).exists() and params.get('overwrite') is False:
            #print(f"{prefix} RELAX EXISTS")
            pose = rosetta_utils.load_pose(dumpfile, from_sequence=False)
            score = sf_fa(pose)
            print(f"{prefix} (precomputed) FastRelax fa_standard score: {score}")
            return {'path': dumpfile, 'score': score, 'model_id': model_id}

        print(f"{prefix} (relax) seceding")
        secede()

        def _setup_movemap(bb=True, chi=False, jump=True):
            mmap = pyrosetta.MoveMap()
            mmap.set_bb(bb)
            mmap.set_chi(chi)
            mmap.set_jump(jump)
            return mmap

        pose = rosetta_utils.load_pose(filename)

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
        print(f"{prefix} (relax) FastRelax fa_standard score: {score}")
        pose.dump_pdb(dumpfile)
        print(f"{prefix} (relax) rejoining")
        rejoin()
        return {'path': dumpfile, 'score': score, 'model_id': model_id}
    except Exception as e:
        with open("except-relax.txt", 'a') as f:
            print(f"args={(filename, input_npz, params)}\nexception={e}", file=f)


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
        params['processes'] = 20
        params['cores'] = 1 
        params['memory'] = '160GB'
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

def make_models(input_npz, output_dir, params):
    """Run the entire modeling workflow
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
    npz = np.load(input_npz, allow_pickle=True)
    seq = "".join(np.atleast_1d(npz['sequence']))
    input_npz = str(input_npz)

    params['TDIR'] = output_dir
    
    # create a models directory
    models = tmpdir.dirname / 'models'
    models.mkdir(exist_ok=True, parents=True)
    params['models'] = str(models)

    # submit centroid model work
    centroid_mgr = Manager(func=minimize)
    for i in range(params['N']):
        mid = f"{i+1:03d}_of_{params['N']}"
        centroid_mgr.add_work((seq, input_npz, {**params, 'model_id': mid}), submit=True)

    centroids = centroid_mgr.results() 
    # adjust the atom dist maximum for restraint setup
    # so that relaxation is more centralized around locality
    #params['ATOM_DIST_MAX'] = 10.0
    
    # record top k centroids
    topk = compute_top_k(centroids, params['K'], str(tmpdir.dirname / 'centroid-models.tsv'))
    print(f"[{output_dir}] Got {len(topk)} top scoring centroid models.")
    relax_mgr  = Manager(func=relax)
    for t in topk:
        relax_mgr.add_work((t['path'], input_npz, {**params, 'model_id': t['model_id']}), submit=True)
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
    print(args.cluster, "is clsuter")
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
                final = make_models(input_npz, output_dir, params)
            else:
                print(f"{input_npz} is finished!")
    
        finals = big_mgr.results()
    finally:
        print("Done!")
        client.shutdown()
