#!/bin/bash

# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

set -euo pipefail

# parameters (can be overriden through environment)
data_dir=${data_dir:-"/scratch/snx3000/dealmeih/ds/mlperf/deepcam/All-Hist/"}
# data_dir="/scratch/snx3000/dealmeih/ds/mlperf/deepcam/deepcam-data-mini/"
output_dir=${output_dir:-"${SCRATCH}/runs/deepcam/rcp"}
local_batch_size=${local_batch_size:-2}
global_batch_size=$(( $local_batch_size * $SLURM_NNODES ))
valid_batch_size=${valid_batch_size:-2}
max_epochs=${max_epochs:-28}
seed=${seed:-42}
run_tag=${run_tag:-"b$(printf '%04d' $global_batch_size)_j$SLURM_JOBID"}

# set learning rate schedule according to RCPs
case $global_batch_size in 
    128)
        LR=0.0010
        lr_warmup_steps=0
        lr_schedule_milestones="8192 16384"
        ;;
    256)
        LR=0.0020
        lr_warmup_steps=0
        lr_schedule_milestones="4096 8192"
        ;;
    512)
        LR=0.0040
        lr_warmup_steps=100
        lr_schedule_milestones="2048 4096"
        ;;
    1024)
        LR=0.0040
        lr_warmup_steps=200
        lr_schedule_milestones="1100 4096"    
        ;;
    2048)
        LR=0.0055
        lr_warmup_steps=400
        lr_schedule_milestones="800"
        ;;
    *)
        echo "No RCP for this global batch size ${global_batch_size} - using default instead."
        LR=$( bc  <<< "0.0000078125 * $global_batch_size" )  # linear (0.001 at bs128)
        LR_STEP0=$( bc <<< "1048576 / $global_batch_size" )  # RCP ([ 8192, 16384 ] at bs128) == Epochs [ 8.35, 16.7 ]
        LR_STEP1=$( bc <<< "2097152 / $global_batch_size" )
        lr_warmup_steps=400
        lr_schedule_milestones="$LR_STEP0 $LR_STEP1"
        ;;
esac

# set MPI environment
export PMI_RANK=$SLURM_NODEID
echo "$PMI_RANK/$SLURM_NTASKS milestones='$lr_schedule_milestones' bz=$global_batch_size lr=$LR warmup=$lr_warmup_steps"


python ./train.py \
       --wireup_method "nccl-slurm-pmi" \
       --wandb_certdir $HOME \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir} \
       --output_dir ${output_dir} \
       --model_prefix "segmentation" \
       --optimizer "LAMB" \
       --start_lr $LR \
       --lr_schedule type="multistep",milestones="${lr_schedule_milestones}",decay_rate="0.1" \
       --lr_warmup_steps ${lr_warmup_steps} \
       --lr_warmup_factor 1 \
       --weight_decay 1e-2 \
       --logging_frequency 10 \
       --save_frequency 0 \
       --max_epochs ${max_epochs} \
       --max_inter_threads 4 \
       --seed ${seed} \
       --batchnorm_group_size 1 \
       --local_batch_size ${local_batch_size} \
       --valid_batch_size ${valid_batch_size} \
       --target_iou 0.82 \
       $@
