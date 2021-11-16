#!/bin/bash
#
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
readonly global_rank=${SLURM_PROCID:-}
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

SLURM_NNODES=${SLURM_NNODES:-$DGXNNODES}
SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
NUMEPOCHS=${NUMEPOCHS:-50}

BATCHSIZE=${BATCHSIZE:-"1"}
LR=${LR:-"0.001"}
WD=${WD:-"0.0"}
INIT_LR=${INIT_LR:-"0.001"}
MOM=${MOM:-"0.9"}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-4}
LRSCHED_EPOCHS=${LRSCHED_EPOCHS:-"32 64"}
LRSCHED_DECAYS=${LRSCHED_DECAYS:-"0.25 0.125"}
DATA_LAYOUT=${DATA_LAYOUT:-"NDHWC"}
DATA_SHARD_TYPE=${DATA_SHARD_TYPE:-"local"}
DATA_SHARD_MULTIPLIER=${DATA_SHARD_MULTIPLIER:-1}

APPLY_LOG_TRANSFORM=${APPLY_LOG_TRANSFORM:-"1"}
APPLY_SHUFFLE=${APPLY_SHUFFLE:-"1"}
APPLY_PRESHUFFLE=${APPLY_PRESHUFFLE:-"1"}
APPLY_PRESTAGE=${APPLY_PRESTAGE:-"1"}
NUM_TRAINING_SAMPLES=${NUM_TRAINING_SAMPLES:-"-1"}
NUM_VALIDATION_SAMPLES=${NUM_VALIDATION_SAMPLES:-"-1"}
DALI_THREADS=${DALI_THREADS:-6}
INSTANCES=${INSTANCES:-1}
USE_H5=${USE_H5:-"1"}
READ_CHUNK_SIZE=${READ_CHUNK_SIZE:-"32"}

# Our HDF5 data is already pre-shuffled. If `USE_H5=1`, setting this
# to 1 has a large performance impact (either only on the staging part
# or on the whole run depending on `APPLY_PRESTAGE`).
APPLY_PRESHUFFLE=$(if [ "$USE_H5" -ge 1 ]; then echo 0; else echo 1; fi)

# Only apply prestaging when we have enough nodes to be able to
# support the memory requirements.
export APPLY_PRESTAGE=$(
    if [ "$(($SLURM_NNODES / $INSTANCES))" -ge 64 ]; then
        echo 1
    else
        echo 0
    fi
       )


PROFILE=${PROFILE:-0}
PROFILE_EXCEL=${PROFILE_EXCEL:-0}
CUDA_PROFILER_RANGE=${CUDA_PROFILER_RANGE:-""}

DTYPE=${DTYPE:-"fp32"}
STATIC_LOSS_SCALE=${STATIC_LOSS_SCALE:-"8192"}
GRAD_PREDIV_FACTOR=${GRAD_PREDIV_FACTOR:-"1.0"}

LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-""}
SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-"/results/checkpoint.data"}

SEED=${SEED:-0}

NUMEXAMPLES=${NUMEXAMPLES:-}
PROFILE_ALL_LOCAL_RANKS=${PROFILE_ALL_LOCAL_RANKS:-0}
THR="0.124"

if [[ ${PROFILE} == 1 ]]; then
    THR="0"
fi

DATAROOT="/data"

echo "running benchmark"
export NGPUS=$SLURM_NTASKS_PER_NODE
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}

if [[ ${PROFILE} -ge 1 ]]; then
    export TMPDIR="/results/"
fi



GPUS=$(seq 0 $(($NGPUS - 1)) | tr "\n" "," | sed 's/,$//')
PARAMS=(
    --data-root-dir "${DATAROOT}"
    --num-epochs "${NUMEPOCHS}"
    --target-mae "${THR}"

    --base-lr "${LR}"
    --initial-lr "${INIT_LR}"
    --momentum "${MOM}"
    --weight-decay "${WD}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --lr-scheduler-epochs $(cat <(for EPOCH in $LRSCHED_EPOCHS; do echo $EPOCH; done))
    --lr-scheduler-decays $(cat <(for DECAY in $LRSCHED_DECAYS; do echo $DECAY; done))

    --training-batch-size "${BATCHSIZE}"
    --validation-batch-size "${BATCHSIZE}"
    --training-samples "${NUM_TRAINING_SAMPLES}"
    --validation-samples "${NUM_VALIDATION_SAMPLES}"

    --data-layout "${DATA_LAYOUT}"
    --data-shard-multiplier "${DATA_SHARD_MULTIPLIER}"
    --dali-num-threads "${DALI_THREADS}"
    --shard-type "${DATA_SHARD_TYPE}"
    --seed "${SEED}"

    --grad-prediv-factor "${GRAD_PREDIV_FACTOR}"
    --instances "${INSTANCES}"

    --load-checkpoint "${LOAD_CHECKPOINT}"
    --save-checkpoint "${SAVE_CHECKPOINT}"
)

if [[ ${APPLY_LOG_TRANSFORM} -ge 1 ]]; then
        PARAMS+=(
            --apply-log-transform
        )
fi

if [[ ${APPLY_SHUFFLE} -ge 1 ]]; then
        PARAMS+=(
            --shuffle
        )
fi

if [[ ${APPLY_PRESHUFFLE} -ge 1 ]]; then
        PARAMS+=(
            --preshuffle
        )
fi

if [[ ${APPLY_PRESTAGE} -ge 1 ]]; then
        PARAMS+=(
            --prestage
        )
fi

if [[ ${USE_H5} -ge 1 ]]; then
        PARAMS+=(
            --use_h5
            --read_chunk_size ${READ_CHUNK_SIZE}
        )
fi

if [[ ${DTYPE} == "amp" ]]; then
    PARAMS+=(
        --use-amp
    )
elif [[ ${DTYPE} == "fp16" ]]; then
    PARAMS+=(
        --use-fp16
        --static-loss-scale ${STATIC_LOSS_SCALE}
    )
fi

# If numexamples is set then we will override the numexamples
#if [[ ${NUMEXAMPLES} -ge 1 ]]; then
#        PARAMS+=(
#        --num-examples "${NUMEXAMPLES}"
#        )
#fi

if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; nothing to do
  DISTRIBUTED=
else
  # Mode 2: Single-node Docker; need to launch tasks with mpirun
  DISTRIBUTED="mpirun --allow-run-as-root --bind-to none --np ${DGXNGPU}"
fi

PROFILE_COMMAND=""
if [[ ${PROFILE} -ge 1 ]]; then
    if [[ ${global_rank} == 0 ]]; then
        if [[ ${local_rank} == 0 ]] || [[ ${PROFILE_ALL_LOCAL_RANKS} == 1 ]]; then
            PROFILE_COMMAND="nsys profile --trace=cuda,nvtx --force-overwrite true --export=sqlite --output /results/${NETWORK}_b${BATCHSIZE}_%h_${local_rank}_${global_rank}.qdrep "
            PARAMS+=(
                --profile
            )

            if [[ ${CUDA_PROFILER_RANGE} != "" ]]; then
                PARAMS+=(
                    --cuda-profiler-range ${CUDA_PROFILER_RANGE}
                )
                PROFILE_COMMAND="${PROFILE_COMMAND} --capture-range cudaProfilerApi  --stop-on-range-end true"
            fi
        fi
    fi
fi
################################################################################
# Binding
################################################################################

if [ -n "${SLURM_CPU_BIND_USER_SET}" ]; then
    echo "Using bindings from SLURM: ${SLURM_CPU_BIND_TYPE}"
    BIND_CMD=""
else
    echo "Using NUMA binding"
    if [ "$TRAINING_SYSTEM" == "booster" ]
      then
        BIND="bash ${SCRIPT_DIR}bind.sh --cpu=${SCRIPT_DIR}juwels_binding.sh \
                  --mem=${SCRIPT_DIR}juwels_binding.sh --ib=single"
    else
      # this is the horeka case
      BIND="bash ${SCRIPT_DIR}bind.sh --cpu=${SCRIPT_DIR}horeka_binding.sh \
                --mem=${SCRIPT_DIR}horeka_binding.sh --ib=single"
    fi
    #BIND_CMD="./bind.sh --cluster=selene --ib=single --cpu=exclusive"
fi

################################################################################
# End binding
################################################################################

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # TODO(ahmadki): track the apiLog.sh bug and remove the workaround
  # there is a bug in apiLog.sh preventing it from collecting
  # NCCL logs, the workaround is to log a single rank only
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

if [[ ${PROFILE} -ge 1 ]]; then
    TMPDIR=/results ${DISTRIBUTED} ${BIND} ${PROFILE_COMMAND} python train.py "${PARAMS[@]}"; ret_code=$?
else
    ${LOGGER:-} ${DISTRIBUTED} ${BIND} python "${COSMOFLOW_DIR}"train.py "${PARAMS[@]}"; ret_code=$?
fi

if [[ ${PROFILE} -ge 1 ]]; then
    if [[ ${PROFILE_EXCEL} -ge 1 ]]; then
        for i in $( find /results -name '*.sqlite' );
        do  
            echo "$(dpkg -l | grep cudnn)"
            echo "python qa/plot_nsight.py --in_files ${i} --out_file "${i/sqlite/txt}" --framework FW_MXNET --prune conv0"
            python qa/plot_nsight.py --in_files ${i} --out_file "${i/sqlite/txt}" --framework FW_MXNET --prune conv0
            echo "python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch ${BATCHSIZE}"
            python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch "${BATCHSIZE}" --mxnet 21.03 --network ${NETWORK}
            python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch "${BATCHSIZE}" --mxnet 20.10 --network ${NETWORK}
            #python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch "${BATCHSIZE}" --filter_type BatchNorm
            python qa/draw_iter_time.py --ifile ${i/sqlite/csv} --max_width 40 --start_index 2 --scale 5 --end_index 90 --batch "${BATCHSIZE}"
        done 
    fi
fi

sleep 3

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="COSMOFLOW_HPC"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
export PROFILE=0
