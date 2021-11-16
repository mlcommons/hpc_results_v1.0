#!/bin/bash

APPDIR=../
cd $APPDIR

####################
N_NODES=$3
master_node=$1
master_addr=$2

worker_num=$(($N_NODES))
RANKS_PER_NODE=8
let N_RANKS=${RANKS_PER_NODE}*${N_NODES}

################

rank=`env | grep OMPI_COMM_WORLD_RANK | awk -F= '{print $2}'`
node_id=`env | grep OMPI_COMM_WORLD_RANK | awk -F= '{print $2}'`
local_rank=`env | grep OMPI_COMM_WORLD_LOCAL_RANK | awk -F= '{print $2}'`  

LOGDIR=../logs
export OMP_NUM_THREADS=1
sleep 1

pin_memory="--pin_memory"
seed=`date +%s`

export NCCL_DEBUG=INFO

id=tgpu-N$N_NODES

#export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices"
CUDA_VISIBLE_DEVICES=$local_rank python main.py --mode train --seed ${seed} --config-yml configs/mlperf_thetagpu.yml --identifier $id --distributed --amp > logs/run-N$N_NODES.log 2>&1 





