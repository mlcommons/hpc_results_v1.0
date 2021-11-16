#!/bin/bash
#SBATCH --nodes=1 -A jucha --partition booster --gres gpu:4 --time=01:00:00
ml purge
ml GCCcore/.9.3.0 GCC/9.3.0 OpenMPI/4.1.0rc1 Python/3.8.5 HDF5/1.10.6 h5py/2.10.0-Python-3.8.5
for x in 2 4 8 16 32; do
    echo "running $x"
    export HDF5_USE_FILE_LOCKING=FALSE
    srun --tasks-per-node=$x  python ./hdf5_benchmark.py
#    srun --tasks-per-node=$x  --jobid $SLURM_JOB_ID python ./hdf5_benchmark.py
done
