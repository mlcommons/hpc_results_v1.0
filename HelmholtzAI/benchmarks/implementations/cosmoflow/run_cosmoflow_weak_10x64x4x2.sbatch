#!/bin/bash

export INSTANCES=10
NODES_PER_INSTANCE=64
RESULTS_ROOT="/p/scratch/jb_benchmark/cosmoflow/results_weak/$INSTANCES/$NODES"
mkdir -p "$RESULTS_ROOT"

./start_training_run.sh --system booster \
    --nodes "$((INSTANCES * NODES_PER_INSTANCE))" \
    --time 02:00:00 \
    --config cosmoflow/configs/config_DGXA100_32x8x2_handrolled.sh
