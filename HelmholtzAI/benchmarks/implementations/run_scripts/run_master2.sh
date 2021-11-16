#!/bin/bash

for i in {1..3}; do
    export RESERVATION="mlperf"
    ./run_jb2048_gpfs.sh
done

export RESERVATION="mlperf"
./run_jb6x512_gpfs.sh

export RESERVATION="mlperf"
./run_jb24x128_gpfs.sh

export RESERVATION="mlperf"
./run_jb24x128_hpst.sh
