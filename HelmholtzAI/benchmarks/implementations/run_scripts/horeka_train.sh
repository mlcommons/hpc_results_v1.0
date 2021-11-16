#!/bin/bash
#
# runs benchmark and reports time to convergence
# this should be run with srun and singularity (see start_jb_training.sh)

# ====================== HoreKa specific settings ============================
export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export NCCL_SOCKET_IFNAME="ib0"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export DGXNGPU=2
export DGXNSOCKET=2
export DGXSOCKETCORES=36 # 76 CPUs / DGXNSOCKET
export DGXHT=2  # HT is on is 2, HT off is 1

export TRAIN_DATA_PREFIX="/data/All-Hist"
export OUTPUT_DIR="/work/run-logs/"

export PROJ_LIB="/opt/conda/share/proj/"
#export PYTHONPATH=/opt/conda/bin/:${PYTHONPATH}
# =============== end of HoreKa specific settings ============================

# network params
PARAMS=(
       --wireup_method="nccl-slurm"
       --run_tag="${SLURM_JOBID}"
       --output_dir="${OUTPUT_DIR}}"
#       --checkpoint                          "None"
       --data_dir_prefix="${TRAIN_DATA_PREFIX}"
#       --max_inter_threads                   "1"
       --max_epochs="200"
       --save_frequency="400"
       --validation_frequency="200"
       --max_validation_steps="50"
       --logging_frequency="1"  # og: 0
       --training_visualization_frequency="200"
       --validation_visualization_frequency="40"
       --local_batch_size"2"
#       --channels                            "{[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}"
       --optimizer="LAMB"
       --start_lr="1e-3"
#       --adam_eps                            "1e-8"
       --weight_decay="1e-2"
#       --loss_weight_pow                     "-0.125"
       --lr_warmup_steps="0"
       --lr_warmup_factor                    "1.0"
       --lr_schedule                         "type=multistep,milestones=800,decay_rate=0.1"
#       "{\"type\":\"multistep\",\"milestones\":\"15000 25000\",\"decay_rate\":\"0.1\"}"
#       --target_iou                          "0.82"
       --model_prefix                        "classifier"
       --amp_opt_level                       "O1"
#       --enable_wandb
#       --resume_logging
#       |& tee
#       -a ${output_dir}/train.out
)

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
readonly global_rank=${SLURM_PROCID:-}
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

echo "running benchmark"
export NGPUS=$SLURM_NTASKS_PER_NODE
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}

if [[ ${PROFILE} -ge 1 ]]; then
    export TMPDIR="/profile_result/"
fi

#GPUS=$(seq 0 $(($NGPUS - 1)) | tr "\n" "," | sed 's/,$//')

# these parameters are taken from the run scripts from other model runs
# commented out parameters are the defaults, can be anabled here

# TODO: fix the profiler for JB (not using right now)
PROFILE_COMMAND=""
if [[ ${PROFILE} == 1 ]]; then
    if [[ ${global_rank} == 0 ]]; then
        if [[ ${local_rank} == 0 ]] || [[ ${PROFILE_ALL_LOCAL_RANKS} == 1 ]]; then
            PROFILE_COMMAND="nvprof --profile-child-processes --profile-api-trace all --demangling on --profile-from-start on  --force-overwrite --print-gpu-trace --csv --log-file /results/rn50_v1.5_${BATCHSIZE}.%h.%p.data --export-profile /results/rn50_v1.5_${BATCHSIZE}.%h.%p.profile "
        fi
    fi
fi

if [[ ${PROFILE} == 2 ]]; then
    if [[ ${global_rank} == 0 ]]; then
        if [[ ${local_rank} == 0 ]] || [[ ${PROFILE_ALL_LOCAL_RANKS} == 1 ]]; then
        PROFILE_COMMAND="nsys profile --trace=cuda,nvtx --force-overwrite true --export=sqlite --output /results/${NETWORK}_b${BATCHSIZE}_%h_${local_rank}_${global_rank}.qdrep "
        fi
    fi
fi

#DISTRIBUTED="srun --mpi=pspmix --cpu-bind=none "
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; nothing to do
  DISTRIBUTED=
else
  # Mode 2: Single-node Docker; need to launch tasks with mpirun
  DISTRIBUTED="mpirun --allow-run-as-root --bind-to none --np ${NGPU}"
fi

# TODO: should the singularity be launched here or with sbatch
if [[ ${PROFILE} -ge 1 ]]; then
    TMPDIR=/results ${DISTRIBUTED} ${PROFILE_COMMAND} python3 train_imagenet.py "${PARAMS[@]}"; ret_code=$?
else
    ${DISTRIBUTED} python /work/src/deepCam/train_hdf5_ddp.py "${PARAMS[@]}"; ret_code=$?
fi

#sleep 3

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
export PROFILE=0
