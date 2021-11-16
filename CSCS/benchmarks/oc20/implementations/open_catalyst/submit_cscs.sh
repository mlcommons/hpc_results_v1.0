#!/bin/bash

source ~/.bashrc
conda activate ocp

export CUDA_VISIBLE_DEVICES=0
export PMI_NO_PREINITIALIZE=1

nodes=${1:-2}
seed=${2:-42}
id=ocp-n$nodes-`date +'%y%m%d-%H%M%S'`

set -x
python main.py --config-yml configs/mlperf_cscs_n$nodes.yml \
    --mode train --distributed --submit \
    --seed $seed \
    --identifier $id \
    --num-nodes $nodes \
    --slurm-timeout 8 \
    --run-dir=$SCRATCH/runs/mlperf/ocp/$id \
    --logdir=$SCRATCH/runs/mlperf/ocp
    # --amp
