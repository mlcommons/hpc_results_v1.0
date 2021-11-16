#!/bin/bash
export TRAINING_INSTANCE_SIZE=3520
export STAGE_DIR_PREFIX=/tmp/deepcam
export STAGE_ONLY=0
#export DATA_DIR_PREFIX="/p/ime-scratch/fs/jb_benchmark/deepCam2/"
CONFIG=configs/best_configs/config_DGXA100_3520GPU_BS3520_graph.sh
./start_training_run.sh -s booster -N $((TRAINING_INSTANCE_SIZE/4)) -c $CONFIG  -t 00:12:00
