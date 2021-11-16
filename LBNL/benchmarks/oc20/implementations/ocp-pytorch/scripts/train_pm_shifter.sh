#!/bin/bash
#SBATCH -C gpu
#SBATCH -J ocp-pm
#SBATCH -A nstaff_g
#SBATCH -q early_science
#SBATCH --image=sfarrell/mlperf-ocp:latest
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time 4:00:00
#SBATCH -o logs/slurm-%x-%j.out

args=$@

# Default settings
: "${OCP_CONFIG:=configs/mlperf_hpc_pm.yml}"

# Distributed config
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29504
export NCCL_SOCKET_IFNAME=hsn
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Run the dummy cuda app to "fix" cuda init errors
if [ ! -f ./dummy ]; then
    echo "int main() {cudaFree(0);}" > dummy.cu && nvcc -o dummy dummy.cu
fi
srun ./dummy

# Using nvidia's bind command requires disabling default cpu bind
if [ "${ENABLE_NV_BINDING:-0}" -eq 1 ]; then
    BIND_SETTINGS="--cpu-bind=none"
fi

set -x
srun -l -u $BIND_SETTINGS shifter scripts/run_training.sh --config-yml $OCP_CONFIG $args
