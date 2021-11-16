# Cosmoflow benchmark

See benchmarks/cosmoflow/implementations/mxnet/README.md for instructions on
acquiring and formatting the input dataset in preparation for running.

To run:

```bash
export CONT=mlperf-cosmoflow:v1.0
export LOGDIR=<directory for output>
export DATADIR=<root directory for the dataset>
source configs/config_DGXA100_128x8x1.sh
sbatch -N ${DGXNNODES} -t ${WALLTIME} ./run.sub
```
