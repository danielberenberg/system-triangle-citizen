# model-from-constraints
Generates a 3D structural model of a protein sequence given structural constraints.
- `extract_constraints.py` will generate constraint matrices from protein structure and save as a `.npz` file.
- `dispatch_mkmod.py` will process constraint `.npz` files and generate models using them. 
- `cst_toolbox/cst_toolbox/mkmod.py` will generate a single decoy or relax a single model.

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

# Technical Details
- Distributed and/or locally multithreaded.
- Tested on SLURM clusters.

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
