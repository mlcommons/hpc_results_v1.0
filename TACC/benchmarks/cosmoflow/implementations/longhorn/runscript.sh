#!/bin/bash
#SBATCH -J cosmoflow
#SBATCH -o %x-%j.out
#SBATCH -A A-ccsc
#SBATCH -N 32
#SBATCH -p v100
#SBATCH -t 6:00:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 10
#SBATCH --sockets-per-node 2
#SBATCH --cores-per-socket 20
##SBATCH --threads-per-core 4
##SBATCH --gres gpu:v100:4
##SBATCH --export=ALL

isDryRun=0    # Run a small case
ppn=4
ranks=$(( $SLURM_NNODES * $ppn ))
echo "SLURM_NNODES =$SLURM_NNODES ppn=$ppn"
rm core*

module reset
ml gcc/7.3.0  
ml python3/powerai_1.7.0 cuda/10.0  nccl/2.4.8 cudnn/7.6.5 tensorflow-py3/2.1.0

mystage="--stage-dir /tmp/cosmo"
mygraph="--wandb"
wandb offline

dirname=result-N$SLURM_NNODES-n$SLURM_NTASKS-j$SLURM_JOBID
tag="--run-tag N$SLURM_NNODES-n$SLURM_NTASKS-j$SLURM_JOBID"

if [[ "$isDryRun" == "0"  ]]
then
    echo "Running Production Run"
	mkdir resultsdir 
    myconfig="configs/cosmo.yaml"
    myoutdir="--output-dir resultsdir/$dirname"
fi

if [[ "$isDryRun" == "1"  ]]
then
   echo "Running DryRun"
   mkdir dryresultsdir
   mytrain="--n-train 32"
   myvalid="--n-valid 32"
   myconfig="configs/cosmo.yaml"
   myepochs="--n-epochs 10"
   mybatchsize="--batch-size 1"
   myoutdir="--output-dir dryresultsdir/$dirname"
fi


echo "Starting train_cgpu.sh at `date`"
SECONDS=0
set -x
time ibrun python3 train.py -d --rank-gpu --mlperf $tag $mytrain $myvalid $mybatchsize $myepochs $mystage $myoutdir $mygraph $myconfig $@
set +x
duration=$SECONDS
echo "finished at `date`"
echo "Run finished for in $duration seconds"
echo "`date` = $(($duration / 3600)) hours : $(($duration / 60)) minutes : $(($duration % 60)) seconds"
