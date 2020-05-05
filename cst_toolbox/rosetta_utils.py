#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys, os
import random
import secrets
import functools
import itertools
from enum import Enum
from pathlib import Path

# numeric/scientific
import numpy as np
from scipy.spatial import distance

import pyrosetta


class ConstraintInfo(object):
    def __init__(self, representation, formatter, symmetric=True):
        self.rep   = representation
        self.templ = formatter
        self.symmetric = symmetric

    def format(self, **slots):
        return self.templ.format(**slots)

    def __str__(self):
        return self.rep
    
    def __repr__(self):
        return self.rep

class ConstraintTypes(Enum):
    dist  = ConstraintInfo(u"\u03B4",
                          "AtomPair CA {RES1} CA {RES2} GAUSSIANFUNC {VALUE} {ATOM_DIST_STD} TAG",
                          symmetric=True)

                                
    dist_ca = ConstraintInfo("C"+u"\u03B1",
                            "AtomPair CA {RES1} CA {RES2} GAUSSIANFUNC {VALUE} {ATOM_DIST_STD} TAG",
                            symmetric=True)

    #dist_cb = ConstraintInfo("C"+u"\u03B2",
    #                        "AtomPair CB {RES1} CB {RES2} GAUSSIANFUNC {VALUE} {ATOM_DIST_STD} TAG",
    #                        symmetric=True)
    #omega = ConstraintInfo(u"\u03C9",
    #                      "Dihedral CA {RES1} CB {RES1} CB {RES2} CA {RES2} CIRCULARHARMONIC {VALUE} {ANGULAR_STD}",
    #                      symmetric=True)
    #theta = ConstraintInfo(u"\u03B8",
    #                      "Dihedral N {RES1} CA {RES1} CB {RES1} CB {RES2} CIRCULARHARMONIC {VALUE} {ANGULAR_STD}",
    #                      symmetric=False)
    #phi   = ConstraintInfo(u"\u03C6",
    #                      "Angle CA {RES1} CB {RES1} CB {RES2} CIRCULARHARMONIC {VALUE} {ANGULAR_STD}",
    #                      symmetric=False)
    #distogram_ca = ConstraintInfo("p(C"+u"\u03B1)",
    #                             "AtomPair CA {RES1} CA {RES2} SPLINE TAG {SPLINE_FILE} {VALUE} 1.0 1.0 0.5",
    #                             symmetric=True)
    #distogram_cb = ConstraintInfo("p(C"+u"\u03B1)",
    #                             "AtomPair CB {RES1} CB {RES2} SPLINE TAG {SPLINE_FILE} {VALUE} 1.0 1.0 0.5",
    #                             symmetric=True)


class DefaultParams(Enum):
    ATOM_DIST_STD = 2.5 
    ANGULAR_STD   = 1.35
    
    ATOM_DIST_MAX   = 10.0
    SEQRES_DIST_MIN =  4 

def _create_dist_restraints(npz, **params):
    """
    Generate constraints (those defined by ConstraintTypes enum) 
    filtered by various conditions (the defaults of which are 
    found in the DefaultParams enum).
    
    args:
        :npz (dict-like) containing phi, theta, omega, dist[_ca|cb|] 
    returns:
        :(dict) - same keys mapping to Rosetta constraint lines
    """
    rst = {cst.name: [] for cst in ConstraintTypes}
    
    ATOM_DIST_STD   = params.get(DefaultParams.ATOM_DIST_STD.name)   or DefaultParams.ATOM_DIST_STD.value
    ATOM_DIST_MAX   = params.get(DefaultParams.ATOM_DIST_MAX.name)   or DefaultParams.ATOM_DIST_MAX.value
    SEQRES_DIST_MIN = params.get(DefaultParams.SEQRES_DIST_MIN.name) or DefaultParams.SEQRES_DIST_MIN.value
    
    Nres = npz[list(npz.keys())[0]].shape[0]

    for cst_type in ConstraintTypes:
        mat = npz.get(cst_type.name)
        if mat is None or None in mat or mat.shape != (Nres, Nres):
            continue

        i, j = np.where(npz[cst_type.name] <= ATOM_DIST_MAX)
        for ri, rj in zip(i, j):
            if not all((ri + 1 < Nres, rj + 1 < Nres)):
                continue
            if abs(ri - rj) < SEQRES_DIST_MIN:
                continue
            if rj > ri:
                val  = mat[ri, rj]
                if not np.isnan(val):
                    line = cst_type.value.format(RES1=ri+1,
                                                 RES2=rj+1,
                                                 VALUE=val,
                                                 ATOM_DIST_STD=ATOM_DIST_STD) #,ANGULAR_STD=ANGULAR_STD) 
                    rst[cst_type.name].append([ri, rj, val, line]) 
    return rst

        


def _create_all_restraints(npz, **params):
    """
    Generate all constraints.

    Only considering constraints for residue pairs that are predicted to be 
    within 20Å of one another
    
    args:
        :npz (dict-like) containing phi, theta, omega, and dist matrices
    returns:
        :(dict) with Rosetta constraints
    """
    rst = {cst.name: [] for cst in ConstraintTypes}

    ATOM_DIST_STD = params.get(DefaultParams.ATOM_DIST_STD.name) or DefaultParams.ATOM_DIST_STD.value
    ATOM_DIST_MAX = params.get(DefaultParams.ATOM_DIST_MAX.name) or DefaultParams.ATOM_DIST_MAX.value
    ANGULAR_STD   = params.get(DefaultParams.ANGULAR_STD.name) or DefaultParams.ANGULAR_STD.value

    i, j = np.where(npz['dist'] <= ATOM_DIST_MAX) 
    Nres = npz['dist'].shape[0]
    for cst_type in ConstraintTypes:
        mat = npz.get(cst_type.name)
        if mat is None or None in mat or mat.shape != (Nres, Nres):
            print(f"No csts for {cst_type}")
            continue
        if cst_type in [ConstraintTypes.distogram_ca, ConstraintTypes.distogram_cb]:
            continue
        for ri, rj in zip(i, j):
            if not all((ri + 1 < Nres, rj + 1 < Nres)):
                continue
            if rj > ri:
                val  = mat[ri, rj]
                if not np.isnan(val):
                    line = cst_type.value.format(RES1=ri+1, RES2=rj+1, VALUE=val, ATOM_DIST_STD=ATOM_DIST_STD, ANGULAR_STD=ANGULAR_STD) 
                    rst[cst_type.name].append([ri, rj, val, line]) 

                if not cst_type.value.symmetric:
                    val  = mat[rj, ri]
                    if not np.isnan(val):
                        line = cst_type.value.format(RES1=rj+1, RES2=ri+1, VALUE=val, ATOM_DIST_STD=ATOM_DIST_STD, ANGULAR_STD=ANGULAR_STD) 
                        rst[cst_type.name].append([rj, ri, val, line]) 
    return rst

@with_initializer(boot_pyrosetta)
def add_constraints(pose, rst, position_interval, seq, tmpdir, nogly=False):
    """Add constraints to a PyRosetta pose"""
    lines = []
    low, high = position_interval

    # setup a function that evaluates whether an AA is a glycine
    if nogly:
        check_glycine = lambda aa: aa != 'G'
    else:
        check_glycine = lambda aa: True
    for key in rst:
        for resi, resj, _, line in rst[key]:
            if low <= abs(resi - resj) < high and all(map(check_glycine, (seq[resi], seq[resj]))):
                lines.append(line)
    
    # write out constraints
    random.shuffle(lines)
    Path(tmpdir / 'tmp').mkdir(exist_ok=True, parents=True)
    with open(Path(tmpdir) / 'tmp' / f"minimize_{secrets.token_hex(16)}.cst", 'w') as cstfile:
        print(*lines, sep='\n', file=cstfile)
        cst_filename = str(cstfile.name)

    constraints = pyrosetta.rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(cst_filename)
    constraints.add_constraints(True)
    constraints.apply(pose)

def generate_constraints(npz, **params):
    """Generate all restraints"""
    rst = _create_dist_restraints(npz, **params)
    #for cst_type in ConstraintTypes:
    #    print(f"# {str(cst_type.value)} constraints: {len(rst[cst_type.name])}")
    return rst

PYROSETTA_BOOTED = "PYROSETTA_BOOTED"

def boot_pyrosetta(param_string=None):
    """Boot PyRosetta"""
    if param_string is None:
        param_string = ""
    
    booted = os.environ.get(PYROSETTA_BOOTED, "no") ==  "yes"
    if not booted:
        pyrosetta.init(param_string)
        os.environ['PYROSETTA_BOOTED'] = "yes"
    
    return True

def with_initializer(init):
    """
    Ensures one function fire before another
    """
    def predecor(func):
        @functools.wraps(func)
        def decorator(*args, initargs=None, **kwargs):
            init(initargs)
            return func(*args, **kwargs)
        return decorator
    return predecor

@with_initializer(boot_pyrosetta)
def omega(CAi, CBi, CBj, CAj):
    """Computes the 'omega' dihedral angle between 
    `CAi` and `CAj` along the virtual axis connecting
    `CBi` and `CBj`
    
    args:
        :CAi, CAj, CBi, CB - real valued 3d vectors
    returns:
        :(float)
    """
    return pyrosetta.rosetta.numeric.dihedral_degrees(CAi, CBi, CBj, CAj)

@with_initializer(boot_pyrosetta)
def theta(Ni, CAi, CBi, CBj):
    """Computes the 'theta' dihedral angle between residues i, j
    
    args:
        :Ni, CAi, CBi, CBj - real valued 3d vectors
    returns:
        :(float)
    """
    return pyrosetta.rosetta.numeric.dihedral_degrees(Ni, CAi, CBi, CBj)

@with_initializer(boot_pyrosetta)
def phi(CAi, CBi, CBj):
    """Computes the 'phi' angle between residues i and j
    args:
        :CAi, CBi, CBj - real valued 3d vectors
    returns:
        :(float)  
    """
    return pyrosetta.rosetta.numeric.angle_degrees(CAi, CBi, CBj)

AA_THREE_LETTER = "ALA ARG ASN ASP ASX CYS GLU GLN GLX GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
AA_ONE_LETTER   = list("ARNDBCEQZGHILKMFPSTWYV")

AA_THREE_TO_ONE = dict(zip(AA_THREE_LETTER, AA_ONE_LETTER))
AA_ONE_TO_THREE = dict(zip(AA_ONE_LETTER, AA_THREE_LETTER))

class ContextMutator(object):
    """Context manager for applying mutations to a pose and reversing them"""
    def __init__(self, source_aa, target_aa, pose, sequence=None):
        """Constructor. *_aa are one letter codes"""
        assert source_aa in AA_ONE_LETTER and target_aa in AA_ONE_LETTER

        self.seq = sequence or pose.sequence()
        self.src = source_aa
        self.trg = target_aa
        self.pose = pose

    def __enter__(self):
        """Mutate forward"""
        for i, residue in enumerate(self.seq):
            if residue == self.src:
                mut = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(i+1, AA_ONE_TO_THREE[self.trg])
                mut.apply(self.pose)
        return self
    
    def __exit__(self, type, value, traceback):
        """Mutate backward"""
        for i, residue in enumerate(self.seq):
            if residue == self.src:
                mut = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(i+1, AA_ONE_TO_THREE[self.src])
                mut.apply(self.pose)
        return self


@with_initializer(boot_pyrosetta)
def constraint_matrices(pose):
    """
    Generate constraint matrices in one pairwise pass.
    """
    ## Mutate GLY -> ALA to ensure the existence of CB atoms and back again 
    with ContextMutator("G", "A", pose) as cm:
        nres  = pose.residues.__len__()
        omegamat, thetamat, phimat, dca_mat, dcb_mat, dmat = np.zeros((6, nres, nres))
        for i in range(nres):
            for j in range(i + 1, nres):
                Ni, Cai, Cbi = map(pose.residue(i + 1).xyz, ('N', 'CA', 'CB'))
                Nj, Caj, Cbj = map(pose.residue(j + 1).xyz, ('N', 'CA', 'CB'))

                omegamat[i, j] = omegamat[j, i] = omega(Cai, Cbi, Cbj, Caj)
                dmat[i, j] = dmat[j, i] = dca_mat[i, j] = dca_mat[j, i] = distance.euclidean(Cai, Caj) 
                dcb_mat[i, j] = dcb_mat[j, i] = distance.euclidean(Cbi, Cbj) 

                phimat[i, j] = phi(Cai, Cbi, Cbj)
                phimat[j, i] = phi(Caj, Cbj, Cbi)

                thetamat[i, j] = theta(Ni, Cai, Cbi, Cbj) 
                thetamat[j, i] = theta(Nj, Caj, Cbj, Cbi)
        
        thetamat, omegamat, phimat = map(np.deg2rad, (thetamat, omegamat, phimat))

    return {'dist_ca': dca_mat,'dist_cb': dcb_mat, 'dist': dmat, 
            'omega': omegamat, 'phi': phimat, 'theta': thetamat,
            'sequence': pose.sequence()}

@with_initializer(boot_pyrosetta)
def load_pose(string_or_path, from_sequence=False):
    string_or_path = str(string_or_path)
    if from_sequence:
        return pyrosetta.pose_from_sequence(string_or_path, 'centroid')
    else:
        return pyrosetta.pose_from_file(string_or_path)

@with_initializer(boot_pyrosetta)
def load_score_fxn_with_weights(filename):
    filename = str(filename)
    sf = pyrosetta.ScoreFunction()
    sf.add_weights_from_file(filename)
    return sf


@with_initializer(boot_pyrosetta)
def initialize_pose_from_rama(sequence):
    """
    args:
        : sequence (str) - The primary structure of the pose
    returns:
        : (pyrosetta.Pose)
    """
    pose = pyrosetta.pose_from_sequence(sequence)

    mover = pyrosetta.rosetta.protocols.backbone_movers.RandomizeByRamaPrePro()
    mover.apply(pose)

    return pose

    
##### trRosetta imports

def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres):
        phi,psi=random_dihedral()
        pose.set_phi(i,phi)
        pose.set_psi(i,psi)
        pose.set_omega(i,180)

    return(pose)

#pick phi/psi randomly from:
#-140  153 180 0.135 B
# -72  145 180 0.155 B
#-122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral():
    phi=0
    psi=0
    r=random.random()
    if(r<=0.135):
        phi=-140
        psi=153
    elif(r>0.135 and r<=0.29):
        phi=-72
        psi=145
    elif(r>0.29 and r<=0.363):
        phi=-122
        psi=117
    elif(r>0.363 and r<=0.485):
        phi=-82
        psi=-14
    elif(r>0.485 and r<=0.982):
        phi=-61
        psi=-41
    else:
        phi=57
        psi=39
    return(phi, psi)

def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)

#
#def _handle_distogram(constraint_type, mats, **params):
#    """
#    Generate distance restraints from distogram tensor
#    ---
#    Tensor T should be shape (L, L, 37) where
#        T[:,:,36]      = No contact
#        T[:,:, i < 36] = 2.0A + i*0.5A contact resolution
#    """
#    return 
#    MEFF  =  0.0001
#    ALPHA =  1.57   
#    DCUT  = 19.5
#    PCUT  =  0.5
#    EBASE = -0.5
#
#    EREP  = [10.0,3.0,0.5]
#    DREP  = [ 0.0,2.0,3.5]
#
#
#    L = mats.shape[0]
#    X, Y = np.indices((L,L))
#    Z = np.argmax(mats, axis=-1) 
#
#    bins  = np.arange(2, 20, 0.5)
#    nbins = 36
#
#    background = (bins / DCUT)**ALPHA
#
#    
#    prob = np.sum(mats[:,:,5:], axis=-1)
#    bkgr = np.array((bins / DCUT)**ALPHA)
#    attr = -np.log((mats[:,:,5:]+MEFF) / (dist[:,:-1][:,:,None]*bkgr[None, None, :])) + EBASE
#    repu = np.maximum(attr[:,:,0], np.zeros((nres, nres))[:,:,None]+np.array(EREP)[None, None, :])
#    dist = np.concatenate([repu, attr], axis=-1)
#    i, j = np.where(prob > PCUT)
#    ########################################################
#    # dist: 0..20A
#    ########################################################
#    bins = np.array([4.25+DSTEP*i for i in range(32)])
#    prob = np.sum(dist[:,:,5:], axis=-1)
#    bkgr = np.array((bins/DCUT)**ALPHA)
#    attr = -np.log((dist[:,:,5:]+MEFF)/(dist[:,:,-1][:,:,None]*bkgr[None,None,:]))+EBASE
#    repul = np.maximum(attr[:,:,0],np.zeros((nres,nres)))[:,:,None]+np.array(EREP)[None,None,:]
#    dist = np.concatenate([repul,attr], axis=-1)
#    bins = np.concatenate([DREP,bins])
#    i,j = np.where(prob>PCUT)
#    prob = prob[i,j]
#    nbins = 35
#    step = 0.5
#    for a,b,p in zip(i,j,prob):
#        if b>a:
#            name=tmpdir.name+"/%d.%d.txt"%(a+1,b+1)
#            with open(name, "w") as f:
#                f.write('x_axis'+'\t%.3f'*nbins%tuple(bins)+'\n')
#                f.write('y_axis'+'\t%.3f'*nbins%tuple(dist[a,b])+'\n')
#                f.close()
#            rst_line = 'AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f'%('CB',a+1,'CB',b+1,name,1.0,step)
#            rst['dist'].append([a,b,p,rst_line])
#    print("dist restraints:  %d"%(len(rst['dist'])))
#    
#    # 0.5A linearly spaced bins
#    DSTEP = 0.5
#    bins = np.array([4.25+DSTEP*i for i in range(32)])
