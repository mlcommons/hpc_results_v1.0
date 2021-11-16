### CSCS DeepCAM submission

The MLPerf HPC submission on Piz Daint on 128 nodes can be obtained by running the SLURM script `run_cscs.sh` in the implementation directory after `setup.sh` is run, as described in `run_and_time.sh`,

```
seed=${seed_value} sbatch -N128 run_cscs.sh

```
where the `seed_value` is set accordingly.
