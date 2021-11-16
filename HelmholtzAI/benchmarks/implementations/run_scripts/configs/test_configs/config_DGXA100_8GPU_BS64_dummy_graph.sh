#!/bin/bash

# hyperparameters
export LOCAL_BATCH_SIZE=1
export START_LR=0.001
export OPTIMIZER="LAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="8192 16384"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=0
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=2

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-dummy"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# auxiliary parameters
export LOGGING_FREQUENCY=10

# misc args
export ADDITIONAL_ARGS="--enable_graph --disable_comm_overlap"

# system parameters
# NVIDIA
#export DGXNGPU=8
#export DGXNNODES=16
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
#export WALLTIME=02:00:00
# Booster
export DGXNGPU=4
export DGXNNODES=32
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=02:00:00

