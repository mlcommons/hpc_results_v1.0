#!/bin/bash

cd ../deepcam
NCCL_ALGO=Tree NCCL_NTHREADS=512 max_epochs=38 sbatch -N1024 run_cscs.sh --torch_num_threads 8 --train_dataset_sharding --train_dataset_caching --valid_dataset_caching --history_logging
