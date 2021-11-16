#!/bin/bash

set -ex

slurm_reservation="" #" --reservation=MLPerfHPC "
slurm_run_exclusive="" # " --dependency singleton "

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
output_dir_prefix="/scratch/snx3000/${USER}/runs/deepcam/mlperf_throughput/${timestamp}_${HOSTNAME}"

# Commandline parsing
if [ "$#" -lt 2 ]; then
    echo "Error: Expecting at least 2 parameters (got $#): (module|sarus) (weak_scaling|strong_scaling)."
fi

if [ "$1" = "sarus" ]; then
    use_container=true
elif [ "$1" = "module" ]; then
    use_container=false
else
    echo "Error: Use either 'sarus' or 'module' to run the container/module-version for first parameter"
    exit -1
fi

if [ "$2" = "weak_scaling" ]; then
    weak_scaling_dataset=true
elif [ "$2" = "strong_scaling" ]; then
    weak_scaling_dataset=false
else
    echo "Error: Use either 'weak_scaling' or 'strong_scaling' for second parameter."
    exit -1
fi

if [ "$#" -gt 2 ]; then
    min_log_n_ranks=$3
    max_log_n_ranks=$4
fi

if [ "$#" -gt 4 ]; then
    log_global_batch_size_range="$(seq $6 -1 $5)"
fi

# Data set & sbatch script
if ${use_container}; then
    exit -1 # Not yet implemented
    # Sarus data_dir
    data_dir=/root/mlperf/data/deepcam/All-Hist/
    sbatch_script=run_cscs.sh
else
    # Module data_dir
    data_dir=/scratch/snx3000/dealmeih/ds/mlperf/deepcam/All-Hist/
    sbatch_script=run_cscs.sh
fi

# mlperf_run: adjust these limits (remove n_epochs, increase time_limit)
n_epochs=3

log_max_local_batch_size=1 # OOM error for local batch size > 2 on P100

# Node number/global batch size range/weak scaling parameters
if ${weak_scaling_dataset}; then 
    if [ -z ${min_log_n_ranks+x} ]; then             min_log_n_ranks=1;               fi
    if [ -z ${max_log_n_ranks+x} ]; then             max_log_n_ranks=1;               fi
    if [ -z ${log_global_batch_size_range+x} ]; then log_global_batch_size_range=(1); fi

    # Weak scaling (constant per-GPU dataset size, simulating full dataset on 1024/512 GPUs)
    weak_n_train_per_rank=118 #236 
    weak_n_valid_per_rank=14  #29
else # full-scale benchmarking
    if [ -z ${min_log_n_ranks+x} ]; then             min_log_n_ranks=7;               fi
    if [ -z ${max_log_n_ranks+x} ]; then             max_log_n_ranks=10;              fi
    if [ -z ${log_global_batch_size_range+x} ]; then log_global_batch_size_range="$(seq $((${log_max_local_batch_size} + ${max_log_n_ranks})) -1 ${min_log_n_ranks})"; fi

    # Strong scaling (use constant, full dataset)
    strong_n_train=121216    
    strong_n_valid=15158    
fi


env_opts="ALGO=Tree,NTHREADS=512"
# PyTorch/NCCL options
export NCCL_ALGO=Tree
export NCCL_NTHREADS=512
# Sarus/TF 2.3: remove this due to bug in XLA/ptxas
export NCCL_DEBUG=WARN
set +x

# Shell helpers
max() {
   [ "$1" -gt "$2" ] && echo $1 || echo $2
}

min() {
   [ "$1" -lt "$2" ] && echo $1 || echo $2
}

# Scaling of throughput
for log_global_batch_size in ${log_global_batch_size_range[@]}; do
    log_n_ranks_ubound=$( min $max_log_n_ranks $log_global_batch_size )
    log_n_ranks_lbound=$( max $min_log_n_ranks $(( log_global_batch_size - ${log_max_local_batch_size} )) ) # OOM error on GPU
    for log_n_ranks in $( seq $( min $max_log_n_ranks $log_global_batch_size ) -1 $( max $min_log_n_ranks $(( log_global_batch_size - ${log_max_local_batch_size} )) ) ); do
        n_ranks=$(( 2**${log_n_ranks} ))
        local_batch_size=$(( 2**(${log_global_batch_size}-${log_n_ranks}) ))

        set -x
        # mlperf_run: adjust these limits (remove n_epochs, increase time_limit)
        if ${weak_scaling_dataset}; then
            dataset_extra_opts="--train_samples $((${weak_n_train_per_rank} * ${n_ranks})) --valid_samples $((${weak_n_valid_per_rank} * ${n_ranks}))"
            epoch_time_limit_mins=$(printf "%.0f" $( bc -l <<< "(${weak_n_train_per_rank} + ${weak_n_valid_per_rank}) / 0.5 / 60 * 2.5 + 1" ) ) # for sbatch job time limit
        else # strong scaling
            dataset_extra_opts="--train_samples ${strong_n_train} --valid_samples ${strong_n_valid}"
            epoch_time_limit_mins=$(printf "%.0f" $( bc -l <<< "(${strong_n_train} + ${strong_n_valid}) / 0.5 / 60 / ${n_ranks} * 2.5 + 1" ) ) # for sbatch job time limit
        fi
        time_limit_mins=$((${epoch_time_limit_mins} * ${n_epochs} + 10))
        set +x

        dataset_optimization_opts=" --train_dataset_sharding --train_dataset_caching --valid_dataset_caching "
        history_logging_opts=" --history_logging "
        torch_num_threads=" --torch_num_threads 8 "

        #if [[ ${local_batch_size} -eq 2 ]]; then
        # mlperf_run: use gpu-gb${global_batch_size}-n${n_ranks}-sub${instance} and create loop around sbatch command: for instance in $(seq 1 ${n_runs}); do .../ done
        output_dir="${output_dir_prefix}/gpu-n${n_ranks}_batch${local_batch_size}_env${env_opts}"
        mkdir -p ${output_dir}
        set -x
        # mlperf_run remove max_epochs and increase time_limit_mins
        data_dir="${data_dir}" output_dir="${output_dir}" local_batch_size=${local_batch_size} max_epochs=${n_epochs} sbatch ${slurm_reservation} --nodes ${n_ranks} --time ${time_limit_mins} --output "${output_dir}/slurm_%j.log" ${slurm_run_exclusive} "${sbatch_script}" ${dataset_extra_opts} ${dataset_optimization_opts} ${history_logging_opts} ${torch_num_threads}
        set +x
        #fi
    done
done

