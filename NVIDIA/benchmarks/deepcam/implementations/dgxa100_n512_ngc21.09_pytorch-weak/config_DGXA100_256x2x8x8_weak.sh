#!/bin/bash

# hyperparameters
export LOCAL_BATCH_SIZE=8
export START_LR=0.001
export OPTIMIZER="LAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="8192 16384"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=0
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=1
export TRAINING_INSTANCE_SIZE=16

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-numpy"
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=8

# staging parameter
export STAGE_DIR_PREFIX="/scratch"
export STAGE_BATCH_SIZE=8
export STAGE_MODE="global"
export STAGE_VERIFY=0
export STAGE_FULL_DATA_PER_NODE=0
export STAGE_USE_DIRECT_IO=1
export STAGE_NUM_WORKERS=8

# auxiliary parameters
export LOGGING_FREQUENCY=10

# misc args
export ADDITIONAL_ARGS="--enable_jit --enable_graph --disable_comm_overlap"

# system parameters
export DGXNGPU=8
export DGXNNODES=512
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:20:00
