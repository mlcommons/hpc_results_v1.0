#!/bin/bash

set -ex

export OCP_CONFIG=configs/pm_b2048.yml

sbatch -d singleton -J pm-b2048 -n 512 scripts/train_pm_shifter.sh --optim.batch_size=4 --seed=1
sbatch -d singleton -J pm-b2048 -n 512 scripts/train_pm_shifter.sh --optim.batch_size=4 --seed=2
sbatch -d singleton -J pm-b2048 -n 512 scripts/train_pm_shifter.sh --optim.batch_size=4 --seed=3
sbatch -d singleton -J pm-b2048 -n 512 scripts/train_pm_shifter.sh --optim.batch_size=4 --seed=4
sbatch -d singleton -J pm-b2048 -n 512 scripts/train_pm_shifter.sh --optim.batch_size=4 --seed=5
