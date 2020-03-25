# model-from-constraints
Generates a 3D structural model of a protein sequence given structural constraints.

# Table of Contents
[model-from-constraints](#model-from-constraints)
- [Install](#install)
- [Modeling Details](#modeling-details)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)
- [References and external links](#references-and-external-links)

# Install
- Create conda environment
    - `conda create -n cst python=3.6`
    - `conda activate cst`
- Clone the repository
    - `git clone https://github.com/danielberenberg/model-from-constraints/`
- Edit your `.condarc` (default located under `$HOME`) to include the Gray Lab channel for PyRosetta
```yaml
channels:
    - https://{USERNAME}:{PASSWORD}@conda.graylab.jhu.edu
    - defaults
```
- Install dependencies
    - `conda install -y dask dask_jobqueue pyrosetta`
- Install the `cst_toolbox` - a short python module for interfacing with Rosetta and preparing constraints
    - `cd model-from-constraints/ && pip install --upgrade ./cst_toolbox`

# Modeling Details
- Structures are modeled using the `Rosetta` macromolecular modeling suite through the `PyRosetta` interface.
- Constraint matrices are read and applied as additions to the Rosetta score function. 
- Starting from a structure roughly sampled from favored regions of Ramachandran space, a simple protocol
of backbone and sidechain minimization is employed with respect to the provided constraints.
- This protocol is greatly influenced by the work done in Yang, et al (2020). 

Constraint matrices are square real-valued matrices of shape `(L, L)` where `L` is the length of the protein sequence and an element
`v` occupying cell `(i,j)` denotes some constraint on the distance/angular orientation between residues `i` and `j`.

Supported constraint types (all angles are in radians):
 - `C𝛼` distances (symmetric).
 - `C𝛽` distances (symmetric).
 - `ω` dihedrals (symmetric), where `ω(i,j)` is the dihedral angle between residue `i` and `j`'s `C𝛼` atoms from the perspective of the the virtual axis connecting their `C𝛽` atoms.
 - `θ` dihedrals (asymmetric), where `θ(i,j)` is the dihedral angle between residue `i`'s `N` atom and `j`'s `C𝛽` from the perspective of the virtual axis connecting `i`'s `C𝛼` to `j`'s `C𝛽`.  
 - `𝜙` angles (asymmetric), where `𝜙(i,j)` is the angle between residue `i`'s `C𝛼` atom and residue `j`'s `C𝛽` atom from the reference point of `i`'s `C𝛽`. 

 Constraint matrices are stored in a single `.npz` file containing the following keys (mapping to square constraint mats): 
 - `dist_ca` - `C𝛼` 
 - `dist_cb` - `C𝛽`  
 - `omega` - `ω`
 - `theta` - `θ`
 - `phi`   - `𝜙`  
 - `sequence` - the length `L` protein sequence

 Each key must be present in the `.npz` file. Optionally, any constraint key (not `sequence`), may have value `None ` (e.g, `cst_npz['phi'] = None`).

# Technical Details
- Distributed and/or locally multithreaded.
- Tested on SLURM cluster.
- `extract_constraints.py` will generate constraint matrices from protein structure and save as a `.npz` file.
- `dispatch_mkmod.py` will process constraint `.npz` files and generate models using them. 
- `cst_toolbox/cst_toolbox/mkmod.py` will generate a single decoy or relax a single model.


# Dependencies
- `Python 3.6` dependencies:
  - `dask`
  - `dask-jobqueue`
  - `pyrosetta` (Requires Academic or Commercial license, see PyRosetta documentation)
    - Can install using `conda`


# References and external links
```bibtex
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
    url = {https://doi.org/10.1146/annurev.biochem.77.062906.171838},
    eprint = {https://doi.org/10.1146/annurev.biochem.77.062906.171838}
}

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
