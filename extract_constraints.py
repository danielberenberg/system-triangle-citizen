#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract constraints from a pdbfile"""
import argparse
from pathlib import Path

import numpy as np

from toolbox.misc import text_color, exists
from toolbox.rosetta_utils import constraint_matrices, load_pose



def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdb", type=exists, help="Input .pdb file to load.")
    parser.add_argument("output_npz", type=Path, help="Output .npz file to write.")
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    pose = load_pose(args.input_pdb, from_sequence=False)

    print(f"Loaded {text_color.LT_RED}{args.input_pdb}{text_color.ENDC}.")
    mats = constraint_matrices(pose)
    np.savez_compressed(args.output_npz, **mats)
    print(f"Wrote out {text_color.GREEN}{args.output_npz}{text_color.ENDC}.")
    
