#!/bin/bash

cd ../deepcam
for seed_value in 1 2 3 5 7 11; do
    seed=${seed_value} sbatch -N128 run_cscs.sh
done
