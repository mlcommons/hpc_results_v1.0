# DeepCAM benchmark - weak scaling

See benchmarks/deepcam/implementations/pytorch/README.md for instructions on
acquiring and formatting the input dataset in preparation for running.

To run:

```bash
export CONT=mplerf-deepcam:v1.0
source ./config_DGXA100_256x2x8x8_weak.sh
sbatch -N${DGXNNODES} -t${WALLTIME} ./run.sub
```
