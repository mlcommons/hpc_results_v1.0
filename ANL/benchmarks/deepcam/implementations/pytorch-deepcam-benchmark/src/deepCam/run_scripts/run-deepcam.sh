#!/bin/bash
#COBALT -n 16
#COBALT -t 4:00:00 
#COBALT -q full-node
#COBALT -A datascience
#COBALT --attrs=pubnet
#COBALT -O n16_submission

echo "Running Cobalt Job $COBALT_JOBID."


#APPDIR=/lus/theta-fs0/projects/datascience/memani/mlhpc/hpc/deepcam/src/deepCam
APPDIR=<source-directory>

## cd to scripts dir
cd $APPDIR/run_scripts

source /etc/profile

module load conda/pytorch
conda activate

COBALT_JOBSIZE=$(cat $COBALT_NODEFILE | wc -l)
N_NODES=${COBALT_JOBSIZE}
ngpus=$((COBALT_JOBSIZE*8))
echo "Running job on ${COBALT_JOBSIZE} nodes"


master_node=$(cat $COBALT_NODEFILE | head -1)
master_addr=$(host $master_node | tail -1 | awk '{print $4}')
worker_num=$(($N_NODES))

export NCCL_IB_DISABLE=0
RANKS_PER_NODE=8
let N_RANKS=${RANKS_PER_NODE}*${N_NODES}

data_dir="/grand/projects/datascience/memani/MLPerf-datasets/deepcam/All-Hist"
output_dir="/grand/projects/datascience/memani/deepcam-output/"
run_tag="test_run"

####################
export PYTHONPATH=/home/memani/.local/conda/pytorch/lib/python3.8/site-packages:$PYTHONPATH

export PYTHONPATH=/lus/theta-fs0/projects/datascience/memani/mlhpc/hpc/deepcam/src/deepCam/mlperf-logging:$PYTHONPATH

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

######################
# run the training

mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $ngpus -hostfile ${COBALT_NODEFILE} --map-by node ./launch.sh $N_NODES


