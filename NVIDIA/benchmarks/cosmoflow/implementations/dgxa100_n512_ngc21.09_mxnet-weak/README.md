# Cosmoflow benchmark - weak scaling

See benchmarks/cosmoflow/implementations/mxnet/README.md for instructions on
acquiring and formatting the input dataset in preparation for running.

To run:

```bash
export CONT=mlperf-cosmoflow:v1.0
export LOGDIR=<directory for output>
export DATADIR=<root directory for the dataset>
source configs/config_DGXA100_32x16x8x1_weak.sh
sbatch -N ${DGXNNODES} -t ${WALLTIME} ./run.sub
```
