[model-from-constraints](#model-from-constraints)
- [Modeling Details](#modeling-details)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)
- [Invoke](#invoke)
- [References and external links](#references-and-external-links)

# model-from-constraints

Generates a 3D structural model of a protein sequence given structural constraints.


# Modeling Details
- Structures are modeled using the `Rosetta` macromolecular modeling suite through the `PyRosetta` interface.
- Constraint matrices are read and applied as additions to the Rosetta score function. 
- Starting from a structure roughly sampled from favored regions of Ramachandran space, a simple protocol
of backbone and sidechain minimization is employed with respect to the provided constraints.
- This protocol is influenced by the work done in Yang, et al (2020). 

# Technical Details
- Distributed and/or locally multithreaded.
- Tested on SLURM clusters.

# Dependencies
- `Python 3.6` dependencies:
  - `dask`
  - `dask-jobqueue`
  - `joblib`    
  - `pyrosetta` (Requires Academic or Commercial license, see PyRosetta documentation)
    - Can install using `conda`

# Invoke
```
usage: make_model.py [-h] [--io-list I_O] [-io I_O] [-N NODES] [-p {ccb,bnl}]
                     [--local] [-n N] [-m K]
                     [-C {dist,omega,phi,theta} [{dist,omega,phi,theta} ...]]
                     [-M {0,1,2}] [--rosetta-init INITARGS]
                     [--score-fxn-wts SCORE_FUNCTION_DIR]

Model structure with npz constraints using PyRosetta.

optional arguments:
  -h, --help            show this help message and exit
  --io-list I_O         Two column (input npz/output dir) filelist
  -io I_O               InputNPZ OutputDirectory
  -N NODES, --cluster-nodes NODES
                        Number of cluster nodes to scale
  -p {ccb,bnl}, --cluster-partition {ccb,bnl}
                        SLURM partition
  --local               Local multithreaded mode
  -n N, --nstruct N     Number of coarse-grained structures to generate
  -m K, --nrelax K      Number (m < n) of structures to perform full-atom
                        refinement
  -C {dist,omega,phi,theta} [{dist,omega,phi,theta} ...], --constraint-types {dist,omega,phi,theta} [{dist,omega,phi,theta} ...]
                        Include only these input channels from the NPZ.
  -M {0,1,2}, --mode {0,1,2}
                        Run mode: 0=short->medium->long, 1=short+medium->long,
                        2=short+medium+long
  --rosetta-init INITARGS
                        Flags to pass to pyrosetta.init (wrap this in doub.
                        quotes).
  --score-fxn-wts SCORE_FUNCTION_DIR
                        Score function weights directory. (Default=data/sfxn)

```

# References and external links
- ```bibtex
@article {Yang1496,
    author = {Yang, Jianyi and Anishchenko, Ivan and Park, Hahnbeom and Peng, Zhenling and Ovchinnikov, Sergey and Baker, David},
    title = {Improved protein structure prediction using predicted interresidue orientations},
    volume = {117},
    number = {3},
    pages = {1496--1503},
    year = {2020},
    doi = {10.1073/pnas.1914677117},
    publisher = {National Academy of Sciences},
    issn = {0027-8424},
    URL = {https://www.pnas.org/content/117/3/1496},
    eprint = {https://www.pnas.org/content/117/3/1496.full.pdf},
    journal = {Proceedings of the National Academy of Sciences}
}
```
- ```bibtex 
@article{doi:10.1146/annurev.biochem.77.062906.171838,
author = {Das, Rhiju and Baker, David},
title = {Macromolecular Modeling with Rosetta},
journal = {Annual Review of Biochemistry},
volume = {77},
number = {1},
pages = {363-382},
year = {2008},
doi = {10.1146/annurev.biochem.77.062906.171838},
    note ={PMID: 18410248},

URL = { 
        https://doi.org/10.1146/annurev.biochem.77.062906.171838
    
},
eprint = { 
        https://doi.org/10.1146/annurev.biochem.77.062906.171838
}
}
```
- ```bibtex
@article{10.1093/bioinformatics/btq007,
    author = {Chaudhury, Sidhartha and Lyskov, Sergey and Gray, Jeffrey J.},
    title = "{PyRosetta: a script-based interface for implementing molecular modeling algorithms using Rosetta}",
    journal = {Bioinformatics},
    volume = {26},
    number = {5},
    pages = {689-691},
    year = {2010},
    month = {01},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btq007},
    url = {https://doi.org/10.1093/bioinformatics/btq007},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/26/5/689/561368/btq007.pdf},
}
```

- <a href="https://github.com/gjoni/trRosetta">trRosetta code</a>
- <a href="https://www.rosettacommons.org/">Rosetta Commons</a>
- <a href="http://www.pyrosetta.org/">PyRosetta homepage</a>
