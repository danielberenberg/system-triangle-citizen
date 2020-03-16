#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import celery
import pyrosetta

from toolbox import rosetta_utils

try:
    print("Using a job celery config")
    import jobceleryconfig
    app = celery.Celery()
    app.config_from_object(jobceleryconfig)
except:
    print("Using a local celery config")
    import celeryconfig
    app = celery.Celery()
    app.config_from_object(celeryconfig)

MinMover = pyrosetta.rosetta.protocols.minimization_packing.MinMover

def _setup_minmover(mmap, fxn, 
                    opt='lbfgs_armijo_nomonotone',
                    cutoff=0.0001,
                    max_iter=1000, cartesian=None):

    # movemap, score function, optimizer, convegence cutoff
    mover = MinMover(mmap, fxn, opt, cutoff, True)
    mover.max_iter(max_iter)
    if cartesian is not None:
        mover.cartesian(cartesian)
    return mover

def _setup_movemap(bb=True, chi=False, jump=True):
    mmap = pyrosetta.MoveMap()
    mmap.set_bb(bb)
    mmap.set_chi(chi)
    mmap.set_jump(jump)
    return mmap
    
@app.task
@rosetta_utils.with_initializer(rosetta_utils.boot_pyrosetta)
def minimize(seq, rst, **params):
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

    #####=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-##### 
    #| setup ScoreFunctions and Mover objects |#
    #####=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-##### 
    weights = sorted(Path(params['score-function-dir']).glob("*.wts"))
    sf, sf1, sf_vdw, sf_cart = map(rosetta_utils.load_score_function_with_weights, weights)     

    mmap = _setup_movemap()
    min_mover      = _setup_minmover(mmap, sf, max_iter=1000) 
    min_mover1     = _setup_minmover(mmap, sf1, max_iter=1000) 
    min_mover_vdw  = _setup_minmover(mmap, sf_vdw, max_iter=500)
    min_mover_cart = _setup_minmover(mmap, sf_cart, cartesisan=True, max_iter=1000)
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
            print("range: ", interval)
            rosetta_utils.add_constraints(pose, rst, interval, seq, params['TDIR'])
            repeat_mover.apply(pose)
            min_mover_cart.apply(pose)
            rosetta_utils.remove_clash(sf_vdw, min_mover1, pose)
    
    centroid_score_function = pyrosetta.create_score_function('cen_std')
    score = centroid_score_function(pose)

    model_id = params.get("model_id") or secrets.token_hex(16)
    dumpfile = str(os.path.join(params['TDIR'], 'centroid_models', "model_" + model_id + '.pdb')) 
    pose.dump_pdb(dumpfile)

    return {'path': dumpfile, 'score': score}


@app.task
@rosetta_utils.with_initializer(rosetta_utils.boot_pyrosetta)
def relax(filename, rst, **params):
    """Encapsulates the relax protocol
    
    args:
        :rst (dict-like) - Generated constraints
    returns:
        :(dict): {'path': str, 'score': score}
    """

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
    switch = SwitchResidueTypeSetMover("fa_standard")
    switch.apply(pose)

    rosetta_utils.add_constraints(pose, rst, (1, len(seq)), pose.sequence(), params['TDIR'], nogly=True)
    relax.apply(pose)

    score = sf_fa(pose)
    model_id = params.get("model_id") or secrets.token_hex(16)
    dumpfile = str(os.path.join(params['TDIR'], 'relax_models', 'final_' + model_id + '.pdb'))
    pose.dump_pdb(dumpfile)

    return {'path': dumpfile, 'score': score}
