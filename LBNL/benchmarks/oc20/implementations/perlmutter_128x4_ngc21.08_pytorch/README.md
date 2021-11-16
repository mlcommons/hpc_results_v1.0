# Perlmutter 128x4x4 PyTorch OpenCatalyst

OpenCatalyst strong-scaling closed-division submission on 128 nodes x 4 GPUs.

To run:

```
export OCP_CONFIG=configs/pm_b2048.yml
sbatch -n 512 scripts/train_pm_shifter.sh
```
