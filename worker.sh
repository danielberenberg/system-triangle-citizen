#!/usr/bin/env sh

#SBATCH --partition bnl
module load slurm

cd /mnt/ceph/users/dberenberg/trRosetta/scripts/constraint-based-modeling
source /mnt/home/dberenberg/anaconda3/etc/profile.d/conda.sh

conda activate trRosetta

python make_model.py -N 2 -p bnl -n 15 -m 5 -io "test/16pkA02.npz test/16pkA02" 

