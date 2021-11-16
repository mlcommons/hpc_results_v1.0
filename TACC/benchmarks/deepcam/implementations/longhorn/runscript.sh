#!/bin/bash
#SBATCH -J Deepcam
#SBATCH -o %x-%j.out
#SBATCH -A A-ccsc
#SBATCH -N 32
#SBATCH -p v100
#SBATCH -t 3:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 10
#SBATCH --sockets-per-node 2
#SBATCH --cores-per-socket 20

ml r tan
conda activate tan

# Setting NCCL Library
# Download NCCL from https://developer.download.nvidia.com/compute/cuda/repos/rhel7/ppc64le/
# if not available on the system
export LD_LIBRARY_PATH=/scratch/05231/aruhela/libs/usr/lib64:$LD_LIBRARY_PATH
export INCLUDE=/scratch/05231/aruhela/libs/usr/include:$INCLUDE

deepdir=/scratch/05231/aruhela/mlperf/hpc/deepcam/src/deepCam
cd $deepdir
rm core*

ppn=4
ranks=$(( $SLURM_NNODES * $ppn ))
echo "SLURM_NNODES =$SLURM_NNODES ppn=$ppn"

local_batch_size=2
output_dir="./${run_tag}"

#data_dir=/scratch/05231/aruhela/newdata/deepcam-data-mini/minidata
data_dir=/scratch/05231/aruhela/newdata/All-Hist

myprintenv

run_tag="deepcam-log-$SLURM_NNODES-$ppn-$local_batch_size-${SLURM_JOBID}"
rm -rf $run_tag

echo "Starting train.py at `date`"
SECONDS=0
set -x	   
/usr/bin/time -f "%E" ibrun \
       python ./train.py \
       --wireup_method "nccl-openmpi" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir} \
       --output_dir ${output_dir} \
       --model_prefix "segmentation" \
       --optimizer "LAMB" \
       --start_lr 0.0055 \
       --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \
       --lr_warmup_steps 400 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --logging_frequency 10 \
       --save_frequency 0 \
       --max_epochs 200 \
       --max_inter_threads 4 \
       --seed $(date +%s) \
       --batchnorm_group_size 1 \
       --local_batch_size ${local_batch_size} 

duration=$SECONDS
echo "Finished at `date`"
echo "Run finished for in $duration seconds"
echo "`date` = $(($duration / 3600)) hours : $(($duration / 60)) minutes : $(($duration % 60)) seconds"

