#!/bin/bash
#COBALT -n 16
#COBALT -t 10:00:00 
#COBALT -q mlperf
#COBALT -A datascience
#COBALT --attrs=pubnet
#COBALT -O n16-submission

echo "Running Cobalt Job $COBALT_JOBID."

APPDIR=/lus/theta-fs0/projects/datascience/memani/mlhpc/hpc/open_catalyst
cd $APPDIR/scripts


source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocp-models

export PATH=/usr/local/cuda-11.0/bin:$PATH
export PATH=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/bin:$PATH
export PYTHONPATH=/grand/projects/datascience/memani/conda/envs/ocp-models/lib/python3.6/site-packages/
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64/:$LD_LIBRARY_PATH


COBALT_JOBSIZE=$(cat $COBALT_NODEFILE | wc -l)
# Notice that we have 8 gpu per node
N_NODES=${COBALT_JOBSIZE}
ngpus=$((COBALT_JOBSIZE*8))
echo "Running job on ${COBALT_JOBSIZE} nodes"


master_node=$(cat $COBALT_NODEFILE | head -1)
master_addr=$(host $master_node | tail -1 | awk '{print $4}')
worker_num=$(($N_NODES))

RANKS_PER_NODE=8
let N_RANKS=${RANKS_PER_NODE}*${N_NODES}


export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export CUDA_DEVICE_ORDER=PCI_BUS_ID


start=`date +%s`

## copy to nvme 
mpirun -n $N_NODES -hostfile ${COBALT_NODEFILE} --map-by node cp -r /grand/projects/datascience/memani/MLPerf-datasets/ocp/oc20_data/s2ef/ /raid/scratch

## run training
mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $ngpus -hostfile ${COBALT_NODEFILE} --map-by node ${mpioptions} ./launch-submission.sh $master_node $master_addr $N_NODES

echo "It took $(($(date +'%s') - $start)) seconds"



