### CSCS DeepCAM submission

The MLPerf HPC submission on Piz Daint on 1024 nodes can be obtained by running the SLURM script `run_cscs.sh` in the implementation directory after `setup.sh` is run, as described in `run_and_time.sh`: 

```
NCCL_ALGO=Tree NCCL_NTHREADS=512 max_epochs=38 sbatch -N1024 run_cscs.sh --torch_num_threads 8 --train_dataset_sharding --train_dataset_caching --valid_dataset_caching --history_logging
```

In particular, this sets up the environment by loading the modules `daint-gpu`, `h5py/3.2.1-CrayGNU-20.11-python3-parallel`, `PyExtensions`, `PyTorch` and runs

```
source env/bin/activate
```

For throughput scaling studies, we use 

```
src/deepCam/run_scripts/submit_mlperf_throughput.sh (module|sarus) (weak_scaling|strong_scaling) log_min_rank log_max_rank log_min_global_batch_size log_max_global_batch_size
```

where the last four parameters define the 2-logarithm of the interval of number of nodes and global batch size that is to be tested (e.g. replace them with 7 10 7 11 to test between 128-1024 nodes and with global batch sizes 128-2048). The second parameter defines if either the dataset should be scaled to a fixed amount per node (weak scaling) or the same fixed dataset should be used for all configurations (strong scaling) and the first if either the module on Piz Daint or the container engine [`Sarus`](https://link.springer.com/chapter/10.1007/978-3-030-34356-9_5) should be used for testing.
