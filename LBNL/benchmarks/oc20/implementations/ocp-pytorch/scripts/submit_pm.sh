#!/bin/bash

set -ex

export OCP_CONFIG=configs/pm_b2048.yml

#sbatch -d singleton -J pm-b2048 -n 256 scripts/train_pm_shifter.sh --optim.batch_size=8
sbatch -x nid002129 -d singleton -J pm-b2048 -n 512 scripts/train_pm_shifter.sh --optim.batch_size=4
#sbatch -x nid002129 -d singleton -J pm-b2048 -n 1024 scripts/train_pm_shifter.sh --optim.batch_size=2
#sbatch -x nid002129 -d singleton -J pm-b2048 -n 2048 scripts/train_pm_shifter.sh --optim.batch_size=1

## Trying for SOTA TTT: b2048 on 2048 gpus
#OCP_CONFIG=configs/pm_b2048.yml \
#    sbatch -J pm-b2048 -n 2048 scripts/train_pm.sh
#
## b4096 on 1024 gpus
#OCP_CONFIG=configs/pm_b4096.yml \
#    sbatch -J pm-b4096 -n 1024 scripts/train_pm_shifter.sh
#
## b2048 on 256 gpus, repeating this run with my old env to check performance
#OCP_CONFIG=configs/pm_b2048.yml \
#    sbatch -J pm-b2048 -n 256 scripts/train_pm.sh
#
## b4096 on 512 gpus
#OCP_CONFIG=configs/pm_b4096.yml \
#    sbatch -J pm-b4096 -n 512 scripts/train_pm_shifter.sh
