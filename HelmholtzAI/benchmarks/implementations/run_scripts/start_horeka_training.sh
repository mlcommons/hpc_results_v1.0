#!/bin/bash
# This file is the first things to be done with srun

ml purge

SRUN_PARAMS=(
  --mpi="pmi2"
  --label
#  --cpu-bind="none"
  --cpus-per-task="4"
  --unbuffered
)
#export DATA_DIR_PREFIX="/hkfs/home/dataset/datasets/deepcam_npy/"
export DATA_DIR_PREFIX="/hkfs/work/workspace/scratch/qv2382-mlperf_data/hdf5s/"
#export STAGE_DIR_PREFIX="${DATA_CACHE_DIRECTORY}"

if [ "${STRIPE_SIZE}" == "tmp" ];
  then
    export STAGE_DIR_PREFIX="/tmp/deepcam"
else
  export STAGE_DIR_PREFIX="/mnt/odfs/${SLURM_JOB_ID}/stripe_${STRIPE_SIZE}"
  export ODFSDIR="${STAGE_DIR_PREFIX}"
fi

export WIREUP_METHOD="nccl-slurm-pmi"
export SEED="0"


export HHAI_DIR="/hkfs/work/workspace/scratch/qv2382-mlperf-submission/hpc_submission_v1.0/HelmholtzAI/"
#"/hkfs/work/workspace/scratch/qv2382-mlperf-combined/MLPerf/"
base_dir="${HHAI_DIR}benchmarks/implementations/"
export DEEPCAM_DIR="${base_dir}image-src/"

SCRIPT_DIR="${base_dir}run_scripts/"
SINGULARITY_FILE="/hkfs/work/workspace/scratch/qv2382-mlperf-combined/MLPerf/benchmarks-closed/deepcam/docker/deepcam_optimized-21.09_2.sif"

export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0

echo "${SINGULARITY_FILE}"

export OUTPUT_ROOT="${HHAI_DIR}results/"
export OUTPUT_DIR="${OUTPUT_ROOT}"

echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"


srun "${SRUN_PARAMS[@]}" singularity exec --nv \
  --bind "${DATA_DIR_PREFIX}","${HHAI_DIR}","${OUTPUT_ROOT}","${DATA_CACHE_DIRECTORY}",/scratch,/tmp ${SINGULARITY_FILE} \
    bash -c "\
      source ${CONFIG_FILE}; \
      export NCCL_DEBUG=INFO; \
      export SLURM_CPU_BIND_USER_SET=\"none\"; \
      bash run_and_time.sh"

