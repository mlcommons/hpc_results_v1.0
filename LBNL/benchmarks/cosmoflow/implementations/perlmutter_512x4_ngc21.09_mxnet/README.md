# Perlmutter 512x4x1 MXNet CosmoFlow

CosmoFlow strong-scaling closed-devision submission on 512 nodes x 4 GPUs.

To run:

```
source configs/config_pm_512x4x1.sh
sbatch -N $DGXNNODES -t $WALLTIME run_pm.sub
```
