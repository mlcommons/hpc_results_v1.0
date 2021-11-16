#!/bin/bash
for i in {1..7}; do
    ./run_jb512_gpfs.sh
done

for i in {1..7}; do
    ./run_jb1024_gpfs.sh
done

for i in {1..7}; do
    export RESERVATION=""
    ./run_jb2048_gpfs.sh
done

#for i in {1..7}; do
#    export RESERVATION="mlperf"
#    ./run_jb2048_gpfs.sh
#done

for i in {1..7}; do
    export RESERVATION="mlperf"
    ./run_jb3520_gpfs.sh
done

export RESERVATION=""
./run_jb12x128_gpfs.sh

export RESERVATION="mlperf"
./run_jb6x512_gpfs.sh
