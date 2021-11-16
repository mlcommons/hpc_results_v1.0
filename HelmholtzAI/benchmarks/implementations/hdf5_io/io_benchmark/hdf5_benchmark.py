from mpi4py import MPI
import h5py
import numpy as np
import time
import pandas as pd
import os

hdf5file="/p/scratch/jb_benchmark/deepCam2/train.h5"
hdf5file="/p/ime-scratch/fs/jb_benchmark/deepCam2/train.h5"
comm=MPI.COMM_WORLD
rank=comm.rank
size=comm.size


tasks_per_node=int(os.environ["SLURM_TASKS_PER_NODE"].split("(")[0])
num_nodes=int(os.environ["SLURM_JOB_NUM_NODES"])

def random_consecutive_benchmark(n_reps, n_subsequent_reads, barrier=False, use_mpio=True):
    durations=[]
    sizes=[]
    if use_mpio:
        f = h5py.File(hdf5file, "r", driver='mpio', comm=comm)
    else:
        f = h5py.File(hdf5file, "r")#, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    ds_data=f.get("data")
    ds_label=f.get("labels")
    for i in range(n_reps):
        start=time.time()
        entry=np.random.randint(ds_data.shape[0]-n_subsequent_reads)
        size=0
        data=ds_data[entry:entry+n_subsequent_reads]
        label=ds_label[entry:entry+n_subsequent_reads]
        size+=data.size*data.dtype.itemsize + label.size*label.dtype.itemsize
        durations.append(time.time()-start)
        sizes.append(size)
        if barrier:
            comm.Barrier()
    f.close()
    return durations, sizes

def get_stats(x, name, pre=""):
    return { name+"_mean": np.mean(x), name + "_max": np.max(x), name+"_min": np.min(x) }

results=[]
for i in range(7):
    n_subsequent_reads=2**i
    n_reps=32//n_subsequent_reads
    n_reps=max(n_reps, 2)
    n_reps=min(n_reps,32)

    durations, sizes=random_consecutive_benchmark(n_reps, n_subsequent_reads, use_mpio=False)
    all_durations=np.array(comm.allgather(durations))
    all_sizes    =np.array(comm.allgather(sizes))
    all_bandwidths=all_sizes/2**30 / all_durations

    if rank==0:
        pre="{:03}".format(n_subsequent_reads)
        this_result={ "n_subsequent_reads": n_subsequent_reads, "tasks_per_node": tasks_per_node, "num_nodes": num_nodes, "mpio": True}
        res=get_stats(all_durations, "duration")
        this_result={**this_result, **res}
        res=get_stats(all_bandwidths, "bw_per_task")
        this_result={**this_result, **res}
        res=get_stats(np.sum(all_bandwidths, axis=0), "bw")
        this_result={**this_result, **res}
        results.append(this_result)

if rank==0:
    df=pd.DataFrame(results)
    print(df.to_string())
    outputfile="hdf5_hpst_random_{:03}_{:03}.csv".format(num_nodes, tasks_per_node) 
    df.to_csv(os.path.join("io_benchmark_results", outputfile) )



