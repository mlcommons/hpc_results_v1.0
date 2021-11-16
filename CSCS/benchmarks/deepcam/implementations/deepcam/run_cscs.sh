#!/bin/bash -l
#SBATCH --job-name=dcam
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --account=csstaff
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --partition=normal

module load daint-gpu h5py/3.2.1-CrayGNU-20.11-python3-parallel PyExtensions PyTorch
module swap cudatoolkit/11.1.0_3.39-4.1__g484e319
source env/bin/activate
# export TORCH_CUDA_ARCH_LIST="6.0"
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

export SLURM_NTASKS=$SLURM_NNODES
export SLURM_LAUNCH_NODE_IPADDR=$(hostname --ip-address)
export CUDA_VISIBLE_DEVICES=0
export PMI_NO_PREINITIALIZE=1

cd src/deepCam; pwd
cat run_scripts/run_cscs.sh

srun -u -l bash run_scripts/run_cscs_rcp.sh $@
