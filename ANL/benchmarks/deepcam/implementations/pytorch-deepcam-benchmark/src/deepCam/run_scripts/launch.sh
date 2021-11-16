#!/bin/bash

APPDIR=/lus/theta-fs0/projects/datascience/memani/mlhpc/hpc/deepcam/src/deepCam/

## cd to scripts dir
cd $APPDIR

####################
N_NODES=$1
worker_num=$(($N_NODES))
################


export NCCL_IB_DISABLE=0

data_dir="/grand/projects/datascience/memani/MLPerf-datasets/deepcam/All-Hist"
output_dir="/grand/projects/datascience/memani/deepcam-output/"
run_tag="test_run-b2-t4"
local_batch_size=2

####################
export PYTHONPATH=/home/memani/.local/conda/pytorch/lib/python3.8/site-packages:$PYTHONPATH

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

mpioptions="-mca pml ob1"
pin_memory="--pin_memory"

######################

local_rank=`env | grep OMPI_COMM_WORLD_LOCAL_RANK | awk -F= '{print $2}'`  
# echo "local rank" $local_rank


CUDA_VISIBLE_DEVICES=$local_rank python ./train.py \
       --wireup_method "nccl-openmpi" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir} \
       --output_dir ${output_dir} \
       --model_prefix "segmentation" \
       --optimizer "LAMB" \
       --start_lr 0.0025 \
       --lr_schedule type="multistep",milestones="4096 8192",decay_rate="0.2" \
       --lr_warmup_steps 100 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --logging_frequency 10 \
       --save_frequency 0 \
       --max_epochs 200 \
       --max_inter_threads 4 \
       --seed $(date +%s) \
       --batchnorm_group_size 1 \
       --local_batch_size ${local_batch_size} >> $APPDIR/logs/deepcam-N$N_NODES.log 2>&1

