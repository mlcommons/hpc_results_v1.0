# DeepCAM benchmark

See benchmarks/deepcam/implementations/pytorch/README.md for instructions on
acquiring and formatting the input dataset in preparation for running.

To run:

```bash
export CONT=mplerf-deepcam:v1.0
source ./config_DGXA100_64x8x2.sh
sbatch -N${DGXNNODES} -t${WALLTIME} ./run.sub
```
