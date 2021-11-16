#!/bin/bash
set -e
#
JOBID=`sbatch  ./sweep_tasks_per_node.sh`
JOBID=`echo $JOBID | cut -d " " -f 4`
echo "JOBID is $JOBID"
for i in 2 4 8 16 32 64 128 256; do
    JOBID=`sbatch -N$i --depend=afterok:${JOBID}  ./sweep_tasks_per_node.sh`
    JOBID=`echo $JOBID | cut -d " " -f 4`
    echo "JOBID is $JOBID"
done
#JOBID=4320244 
#for i in 64 128 256; do
#    JOBID=`sbatch -N$i --depend=afterok:${JOBID}  ./sweep_tasks_per_node.sh`
#    JOBID=`echo $JOBID | cut -d " " -f 4`
#    echo "JOBID is $JOBID"
#done
