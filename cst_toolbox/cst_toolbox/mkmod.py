#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Microservice to generate 1 model"""

# builtin
import os
import json
import secrets
import argparse
from enum import Enum
from pathlib import Path

# third party
import numpy as np

# local
from cst_toolbox import rosetta_utils
from cst_toolbox.misc import exists, WorkingDirectory

os.environ["OPENBLAS_NUM_THREADS"] = "1"

PATH = Path(__file__).absolute()

_DEFAULT_SCORE_FILES = Path(__file__).parent / 'data' / 'sfxn'

if not _DEFAULT_SCORE_FILES.is_dir():
    _DEFAULT_SCORE_FILES = None

ROSETTA_LOGLEVEL = 100
class DefaultArguments(Enum):
    """Default command line parameters"""
    constraint_types   = ['dist', 'theta', 'omega', 'phi']
    score_function_dir = _DEFAULT_SCORE_FILES 
    overwrite          = False
    mode               = 0
    initargs           =  f"-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level {ROSETTA_LOGLEVEL}"

DEFAULT_PARAMS = {param.name: param.value for param in DefaultArguments}

def _setup_arguments(parser, defaults=True):
    parser.add_argument("--model_json", type=Path)
    parser.add_argument("--model_id")

    parser.add_argument("--relax", help="Input PDB file if relax mode", type=exists, dest='input_pdb')

    parser.add_argument("-M", "--mode", choices=[0, 1, 2], dest='mode',
                        type=int, help="Run mode: 0=short->medium->long, 1=short+medium->long, 2=short+medium+long")

    # mode parameters
    parser.add_argument("-C", "--constraint-types", nargs='+',
                        choices=[cst.name for cst in rosetta_utils.ConstraintTypes],
                        dest='constraint_types', 
                        help="Include only these input channels from the NPZ.")

    # Rosetta parameters
    parser.add_argument("--rosetta-init", type=str,
                        dest='initargs',
                        help="Flags to pass to pyrosetta.init (wrap this in doub. quotes).")

    parser.add_argument("--score-fxn-wts", dest='score_function_dir',
                        help=f"Score function weights directory. (Default={DefaultArguments.score_function_dir.value or 'Not found...'})",
                        required=DefaultArguments.score_function_dir.value is None)

    parser.add_argument("--overwrite", help="Overwrite paths", action='store_true', dest='overwrite')
    
    if defaults:
        parser.set_defaults(**DEFAULT_PARAMS)

def arguments():
    parser = argparse.ArgumentParser(description="Make or relax model")

    # add positionals
    parser.add_argument("input_npz", help="Input constraint matrix file", type=exists)
    parser.add_argument("TDIR", help="Output directory", type=exists)

    _setup_arguments(parser)
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

    model_id = params.get("model_id") or secrets.token_hex(16)
    dumpfile = str(os.path.join(params['models'], "model_" + model_id + '.pdb')) 

    prefix = f"[{Path(params['TDIR']).name}-{model_id}]"
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

    #####=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-##### 
    #| setup ScoreFunctions and Mover objects |#
    #####=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-##### 
    sfdir = Path(params['score_function_dir'])
    sfs   = [str(sfdir / sffn) for sffn in ['scorefxn.wts', 'scorefxn1.wts', 'scorefxn_vdw.wts', 'scorefxn_cart.wts']]
    print(sfs)
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

    return {'path': dumpfile, 'score': score, 'model_id': model_id}


def relax(filename, rst , params):
    """Encapsulates the relax protocol
    
    args:
        :rst (dict-like) - Generated constraints
    returns:
        :(dict): {'path': str, 'score': score}
    """
    
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
    return {'path': dumpfile, 'score': score, 'model_id': model_id}


if __name__ == '__main__':
    import pyrosetta

    args = arguments()
    params = vars(args)

    params['models'] = str(Path(params['TDIR']) / 'models')
    Path(params['models']).mkdir(exist_ok=True, parents=True)

    pyrosetta.init(params['initargs'])

    npz = np.load(args.input_npz, allow_pickle=True)

    rst = rosetta_utils.generate_constraints(npz, **params)
    
    relax_mode = args.input_pdb is not None
    
    WorkingDirectory(Path(args.TDIR)).setup(exist_ok=True, parents=True)

    if relax_mode:
        pdb = str(params['input_pdb'])
        X = (pdb, rst, params)
        f = relax
        mode = 'relax'
    else:
        seq = "".join(np.atleast_1d(npz['sequence']))
        X = (seq, rst, params)
        f = minimize
        mode = 'centroid'

    y = f(*X)
    if args.model_json is not None:
        with open(args.model_json, 'w') as js:
            js.write(json.dumps({**y, 'mode': mode}))
