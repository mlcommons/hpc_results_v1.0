#!/bin/bash

# hooray for stack overflow...
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Launcher for training + timing for Cosmoflow on Juwels Booster"
      echo " "
      echo "[options] application [arguments]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-s, --system              the HPC machine to use [booster]"
      echo "-N, --nodes               number of nodes to compute on"
      echo "-t, --time                compute time limit"
      echo "-c, --config              config file to use"
      exit 0
      ;;
    -s|--system) shift; export TRAINING_SYSTEM=$1; shift; ;;
    -N|--nodes) shift; export SLURM_NNODES=$1; shift; ;;
    -t|--time) shift; export TIMELIMIT=$1; shift; ;;
    -c|--config) shift; export CONFIG_FILE=$1; shift; ;;
    *) break; ;;
  esac
done

if [ -z ${TIMELIMIT} ]; then TIMELIMIT="00:10:00"; fi

echo "Job time limit: "${TIMELIMIT}

SBATCH_PARAMS=(
  --nodes              "${SLURM_NNODES}"
  --tasks-per-node     "4"
  --time               "${TIMELIMIT}"
  --gres               "gpu:4"
  --job-name           "cosmoflow-mlperf"
  --time               "${TIMELIMIT}"
)

export TRAINING_SYSTEM="${TRAINING_SYSTEM}"
export STAGING_AREA=/staging_area

if [ "$TRAINING_SYSTEM" == "booster" ]
  then
    hhai_dir="/p/project/jb_benchmark/MLPerf-1.0-combined/MLPerf/"
    export OUTPUT_ROOT="${hhai_dir}results/cosmoflow/"
    export OUTPUT_DIR="${OUTPUT_ROOT}"

    SBATCH_PARAMS+=(
      --partition     "booster"
      --output        "${OUTPUT_DIR}slurm-cosmo-JB-N-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm-cosmo-JB-N-${SLURM_NNODES}-%j.err"
      --account       "hai_mlperf"
    )
    sbatch "${SBATCH_PARAMS[@]}" start_jb_training.sh

elif [ "$TRAINING_SYSTEM" == "horeka" ]
  then
    hhai_dir="/hkfs/work/workspace/scratch/qv2382-mlperf-combined/MLPerf/"
    export OUTPUT_ROOT="${hhai_dir}results/cosmoflow/"
    export OUTPUT_DIR="${OUTPUT_ROOT}"

    SBATCH_PARAMS+=(
      --partition     "accelerated"
      --output        "${OUTPUT_DIR}slurm-cosmo-HoreKa-N-${SLURM_NNODES}-%j.out"
      --error         "${OUTPUT_DIR}slurm-cosmo-HoreKa-N-${SLURM_NNODES}-%j.err"
      --exclude       "hkn[0518,0519,0533,0614,0625,0811]"
      --cpu-freq="high"
      --gpu-freq="high"
      --constraint="BEEOND"
      -A "hk-project-test-mlperf"
    )
    sbatch "${SBATCH_PARAMS[@]}" start_horeka_training.sh
else
  echo "must specify system that we are running on! give as first unnamed parameter"
  exit 128
fi
