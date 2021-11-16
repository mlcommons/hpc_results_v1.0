## converted from NVIDIA's benchmarks from MLPerf 0.7
export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export DALI_DONT_USE_MMAP=0 # 0 for /raid and 1 for lustre

# this shouldnt be needed (we are not using hvd for the closed benchmarks)
export HOROVOD_NUM_NCCL_STREAMS=1
export HOROVOD_CYCLE_TIME=0.1

## System config params
export NCCL_SOCKET_IFNAME="ib0"

export CUDA_VISIBLE_DEVICES="0,1,2,3"

#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# JB
if [ "$TRAINING_SYSTEM" == "booster" ]
  then
    # JB
    export DGXNGPU=2
    export DGXNSOCKET=2
    export DGXSOCKETCORES=24 # 76 CPUs / DGXNSOCKET
    export DGXHT=2  # HT is on is 2, HT off is 1
    export TRAIN_DATA_PREFIX="/p/largedata/datasets/MLPerf/MLPerfHPC/deepcam_v1.0/"
elif [ "$TRAINING_SYSTEM" == "horeka" ]
  then
    # horeka
    export DGXNGPU=2
    export DGXNSOCKET=2
    export DGXSOCKETCORES=36 # 76 CPUs / DGXNSOCKET
    export DGXHT=2  # HT is on is 2, HT off is 1
    export TRAIN_DATA_PREFIX="/hkfs/home/datasets/deepcam/"
else
  echo "must specify system that we are running on! give as first unnamed parameter"
fi

# unclear if needed -> this is a tunable parameter
#bind_cpu_cores=([0]="48-51,176-179" [1]="60-63,188-191" [2]="16-19,144-147" [3]="28-31,156-159"
#                [4]="112-115,240-243" [5]="124-127,252-255" [6]="80-83,208-211" [7]="92-95,220-223")
#
#bind_mem=([0]="3" [1]="3" [2]="1" [3]="1"
#          [4]="7" [5]="7" [6]="5" [7]="5")


#[coquelin1@hdfmll01 mxnet]$ cat config_DGXA100_multi_230x8x35.sh
#source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

#export DALI_PREFETCH_QUEUE="3"
#export DALI_NVJPEG_MEMPADDING="256"
#export DALI_CACHE_SIZE="12288"
#
## Default is no NCCL and BWD overlap
#export HOROVOD_CYCLE_TIME=0.1
#export HOROVOD_FUSION_THRESHOLD=67108864
#export HOROVOD_NUM_NCCL_STREAMS=2
#export MXNET_HOROVOD_NUM_GROUPS=1
#export NHWC_BATCHNORM_LAUNCH_MARGIN=32
#export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
#export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999
#export MXNET_EXPERIMENTAL_ENABLE_CUDA_GRAPH=1
#
### System run parms
#export DGXNNODES=230
#export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
#export WALLTIME=00:30:00

