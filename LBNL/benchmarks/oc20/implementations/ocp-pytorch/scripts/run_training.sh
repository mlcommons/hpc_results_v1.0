#!/bin/bash

# This script will run the actual training command with provided
# command line options. It is run by every rank and sets the per-rank
# environment variables needed for pytorch distributed initialization.

args=$@
id=${SLURM_JOB_NAME}-n${SLURM_NTASKS}-${SLURM_JOB_ID}

export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NVIDIA's binding script
if [ "${ENABLE_NV_BINDING:-0}" -eq 1 ]; then
    if [[ "${NERSC_HOST}" == "perlmutter" ]]; then
        bind_map=scripts/pm_bind_map.sh
        BIND_CMD="./scripts/bind.sh --cpu=$bind_map --mem=$bind_map --"
    else
        BIND_CMD="./scripts/bind.sh --cpu=exclusive --"
    fi
fi

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    PROFILE_DIR=${PROFILE_DIR:-$SCRATCH/ocp/profile}
    NSYS_ARGS=${NSYS_ARGS:---trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true}
    PROFILE_CMD="nsys profile $NSYS_ARGS -o $PROFILE_DIR/${id}"
    mkdir -p $PROFILE_DIR
fi

set -x
${BIND_CMD} ${PROFILE_CMD} python main.py --mode train \
    --distributed \
    --local_rank $LOCAL_RANK \
    --identifier $id $args
