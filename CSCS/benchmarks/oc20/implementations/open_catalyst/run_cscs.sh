#!/bin/bash -l
 
#SBATCH --job-name=ocp
#SBATCH --time=04:00:00
#SBATCH --nodes=3
#SBATCH --account=usup
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu

source ~/.bashrc
conda activate ocp
# srun which python

# export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=ipogif0
# export NCCL_IB_CUDA_SUPPORT=1
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

id=ocp-n$SLURM_NNODES-`date +'%y%m%d-%H%M%S'`
logdir=$SCRATCH/runs/mlperf/ocp
mkdir -p $logdir/$id

ulimit -c 0  # Disable core file creation

export MASTER_ADDR=$(hostname --ip-address)
export MASTER_PORT=13356
export WORLD_SIZE=$SLURM_NNODES
export SRC='RANK=$SLURM_NODEID  python -u main.py --config-yml configs/mlperf_cscs_n${WORLD_SIZE}.yml --mode train --seed 42 --distributed \
    --run-dir=$logdir/$id \
    --seed 42 \
    --identifier $id \
    --slurm-timeout 8 \
    --logdir=$logdir'

srun -N2 -l bash -c "hostname; echo $SRC"
srun -ul bash -c "$SRC"
