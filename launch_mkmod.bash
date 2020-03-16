#!/usr/bin/env bash

#SBATCH --partition bnl
#SBATCH --job-name mkmod 

module load slurm

cd /mnt/ceph/users/dberenberg/trRosetta/scripts/constraint-based-modeling
source /mnt/home/dberenberg/anaconda3/etc/profile.d/conda.sh

conda activate trRosetta

python make_model.py -N 8 --cluster-partition bnl -n 150 -m 50 -io "test/16pkA02.npz test/16pkA02_full_run_distributed"

