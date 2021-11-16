#!/bin/bash
# This file is the first things to be done with srun

ml purge

SRUN_PARAMS=(
  --mpi            pspmix
  --cpu-bind       none
#  --unbuffered
#  --label
)


export SLURM_CPU_BIND_USER_SET="none"

if [ -z $DATA_DIR_PREFIX ]; then
    export DATA_DIR_PREFIX="/p/scratch/jb_benchmark/deepCam2/"
fi
export WIREUP_METHOD="nccl-slurm"
export SEED="0"

hhai_dir="/p/project/jb_benchmark/MLPerf-1.0-combined/MLPerf/"

base_dir="${hhai_dir}benchmarks-closed/deepcam/"
export DEEPCAM_DIR="${base_dir}image-src/"

SCRIPT_DIR="${base_dir}run_scripts/"
export SINGULARITY_FILE="/p/project/jb_benchmark/nvidia_singularity_images/nvidia_deepcam_21.09-pmi2.sif"

export OUTPUT_ROOT="${hhai_dir}results/deepcam2/"
export OUTPUT_DIR="${OUTPUT_ROOT}"

echo "CONFIG_FILE ${CONFIG_FILE}"

echo "${CONFIG_FILE}"
cat "${CONFIG_FILE}"

export UCX_MEMTYPE_CACHE=0
export NCCL_IB_TIMEOUT=20
export SHARP_COLL_LOG_LEVEL=3
export OMPI_MCA_coll_hcoll_enable=0
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_COLLNET_ENABLE=0


srun "${SRUN_PARAMS[@]}" bash -c '
    MASTER=$(echo "$SLURM_STEP_NODELIST" | cut -d "," -f 1);
    singularity run --nv \
  --bind "${DATA_DIR_PREFIX}",${SCRIPT_DIR},${OUTPUT_ROOT},/p/project/jb_benchmark/kesselheim1/MLPerf/benchmarks-closed/deepcam/run_scripts/my_pytorch/distributed_c10d.py:/opt/conda/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py ${SINGULARITY_FILE} \
    bash -c "\
      export CUDA_VISIBLE_DEVICES="0,1,2,3";  \
      export PMIX_SECURITY_MODE="native";
      export NCCL_DEBUG=INFO; \
      export NCCL_DEBUG_SUBSYS=INIT,GRAPH ; \
      source ${CONFIG_FILE}; \
      bash run_and_time.sh"'
