#!/bin/bash
#
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
#if [ "$WORLD_RANK" == 0 ]; then
echo "STARTING TIMING RUN AT $start_fmt"
pwd
echo "CUDA_VISIBLE_DEVICES (start): ${CUDA_VISIBLE_DEVICES}"
#fi
# assemble launch command
export RUN_TAG=${RUN_TAG:-deepcam_ngpu$(( ${SLURM_NNODES} * ${DGXNGPU} ))_${SLURM_JOBID}}
export OUTPUT_DIR=${OUTPUT_ROOT:-/tmp}/${RUN_TAG}

# create tmp directory
mkdir -p ${OUTPUT_DIR}/logs

# LR switch
if [ -z ${LR_SCHEDULE_TYPE} ]; then
    lr_schedule_arg=""
else
    lr_schedule_arg="--lr_schedule type=\"${LR_SCHEDULE_TYPE}\",milestones=\"${LR_MILESTONES}\",decay_rate=\"${LR_DECAY_RATE}\""
fi

PARAMS=(
    --wireup_method ${WIREUP_METHOD}
    --run_tag ${RUN_TAG}
    --data_dir_prefix ${DATA_DIR_PREFIX:-"/data"}
    --output_dir ${OUTPUT_DIR}
    --model_prefix "segmentation"
    --optimizer ${OPTIMIZER}
    --start_lr ${START_LR}
    --lr_schedule type="${LR_SCHEDULE_TYPE}",milestones="${LR_MILESTONES}",decay_rate="${LR_DECAY_RATE}"
    --lr_warmup_steps ${LR_WARMUP_STEPS}
    --lr_warmup_factor ${LR_WARMUP_FACTOR}
    --weight_decay ${WEIGHT_DECAY}
    --logging_frequency ${LOGGING_FREQUENCY}
    --save_frequency 100000
    --max_epochs ${MAX_EPOCHS:-200}
    --max_inter_threads ${MAX_THREADS:-4}
    --seed ${SEED}
    --batchnorm_group_size ${BATCHNORM_GROUP_SIZE}
    --shuffle_mode "${SHUFFLE_MODE}"
    --data_format "${DATA_FORMAT}"
    --data_oversampling_factor ${DATA_OVERSAMPLING_FACTOR:-1}
    --precision_mode "${PRECISION_MODE}"
    --enable_nhwc
    --local_batch_size ${LOCAL_BATCH_SIZE}
    --local_batch_size_validation ${LOCAL_VALIDATION_BATCH_SIZE}
    ${ADDITIONAL_ARGS}
)

# profile command:
if [ -n "${SLURM_PROCID}" ]; then
    WORLD_RANK=${SLURM_PROCID}
elif [ -n "${OMPI_COMM_WORLD_RANK}" ]; then
    WORLD_RANK=${OMPI_COMM_WORLD_RANK}
elif [ -n "${PMIX_RANK}" ]; then
    WORLD_RANK=${PMIX_RANK}
elif [ -n "${PMI_RANK}" ]; then
    WORLD_RANK=${PMI_RANK}
fi

PROFILE_BASE_CMD="nsys profile --mpi-impl=openmpi --trace=cuda,cublas,nvtx,mpi -f true -o ${OUTPUT_DIR}/profile_rank${WORLD_RANK}"
if [[ ${ENABLE_PROFILING} == 1 ]]; then
    if [ "$WORLD_RANK" == 0 ]; then
        echo "Profiling enabled"
    fi
    PROFILE_CMD=${PROFILE_BASE_CMD}
else
    PROFILE_CMD=""
fi

################################################################################
# Binding
################################################################################

if [ -n "${SLURM_CPU_BIND_USER_SET}" ]; then
    if [ "$WORLD_RANK" == 0 ]; then
        echo "Using bindings from SLURM: ${SLURM_CPU_BIND_TYPE}"
    fi
    BIND_CMD=""
else
    echo "Using NUMA binding"
    if [ "$TRAINING_SYSTEM" == "booster" ]
      then
        BIND_CMD="bash ${SCRIPT_DIR}bind.sh --cpu=${SCRIPT_DIR}juwels_binding.sh \
                  --mem=${SCRIPT_DIR}juwels_binding.sh --ib=single"
    else
      # this is the horeka case
      BIND_CMD="bash ${SCRIPT_DIR}bind.sh --cpu=${SCRIPT_DIR}horeka_binding.sh \
                --mem=${SCRIPT_DIR}horeka_binding.sh --ib=single"
    fi
    #BIND_CMD="./bind.sh --cluster=selene --ib=single --cpu=exclusive"
fi

################################################################################
# End binding
################################################################################
pushd "${DEEPCAM_DIR}"

# do we cache data
if [ ! -z "${DATA_CACHE_DIRECTORY}" ]; then
    PARAMS+=(--data_cache_directory "${DATA_CACHE_DIRECTORY}")
fi

# run script selection:
if [ ! -z "${TRAINING_INSTANCE_SIZE}" ]; then
    echo "Running Multi Instance Training"
    RUN_SCRIPT="./train_instance.py"
    PARAMS+=(--training_instance_size "${TRAINING_INSTANCE_SIZE}")

    if [ ! -z "${STAGE_DIR_PREFIX}" ]; then
	PARAMS+=(
	    --stage_dir_prefix ${STAGE_DIR_PREFIX}
	    --stage_num_workers ${STAGE_NUM_WORKERS:-1}
	    --stage_batch_size ${STAGE_BATCH_SIZE:--1}
	    --stage_mode ${STAGE_MODE:-"node"}
	    --data_staging_method ${STAGE_METHOD:-"instance"}
	)
	# do we need to verify the staging results
	if [ "${STAGE_VERIFY:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_verify)
	fi
	if [ "${STAGE_ONLY:-0}" -eq 1 ]; then
	    echo "WARNING: You are about to run a staging only benchmark"
	    PARAMS+=(--stage_only)
	fi
	if [ "${STAGE_FULL_DATA_PER_NODE:-0}" -eq 1 ]; then
	    PARAMS+=(--stage_full_data_per_node)
	fi
    fi
elif [ ! -z ${CAPTURE_RANGE_START} ]; then
    echo "Running Profiling"
    RUN_SCRIPT="./profile.py"
    PARAMS+=(
      --capture_range_start ${CAPTURE_RANGE_START}
      --capture_range_stop ${CAPTURE_RANGE_STOP}
      ${ADDITIONAL_PROFILE_ARGS}
    ) 
else
    echo "Running Single Instance Training"
    RUN_SCRIPT="./train.py"
fi

# cleanup command
CLEANUP_CMD="cp ${OUTPUT_DIR}/logs/${RUN_TAG}.log /results/; \
             sed -i 's|SUBMISSION_ORG_PLACEHOLDER|NVIDIA Corporation|g' /results/${RUN_TAG}.log; \
	           sed -i 's|SUBMISSION_PLATFORM_PLACEHOLDER|${DGXSYSTEM}|g' /results/${RUN_TAG}.log"

# run command
echo "running ${BIND_CMD} ${PROFILE_CMD} python ${RUN_SCRIPT} ${PARAMS[@]}"
${BIND_CMD} ${PROFILE_CMD} python ${RUN_SCRIPT} "${PARAMS[@]}"; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# cleanup command
${CLEANUP_CMD}

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="DEEPCAM_HPC"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
