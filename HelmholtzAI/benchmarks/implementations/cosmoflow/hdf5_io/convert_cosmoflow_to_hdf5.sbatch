#!/usr/bin/env bash

#SBATCH --nodes=4
#SBATCH --tasks-per-node=8
#SBATCH --gres=gpu
#SBATCH --time=23:00:00
#SBATCH --partition=booster
#SBATCH --account=atmlaml

ml GCCcore/.9.3.0 GCC/9.3.0 OpenMPI/4.1.0rc1 Python/3.8.5 HDF5/1.10.6 h5py/2.10.0-Python-3.8.5

srun python convert_cosmoflow_to_hdf5.py
