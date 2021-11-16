import os
from glob import glob
import itertools
import numpy as np
import time
from queue import LifoQueue as Queue
import torch.cuda.nvtx as nvtx
import copy

import h5py
import subprocess


def get_shard_range(num_files, num_shards, shard_id, cycle_dist=0):
    assert (shard_id < num_shards)
    # shard files into bulk and remainder:
    num_files_per_shard = num_files // num_shards
    # num_files_bulk = num_files_per_shard * num_shards
    num_files_remainder = num_files % num_shards

    shard_start = [0]
    for i in range(1, num_shards):
        if i - 1 < num_files_remainder:
            this_shard_start = shard_start[-1] + (num_files_per_shard + 1)
        else:
            this_shard_start = shard_start[-1] + (num_files_per_shard)
        shard_start.append(this_shard_start)
    shard_start.append(num_files)

    ranges = []
    for i in range(num_shards):
        ranges.append((shard_start[i], shard_start[i + 1]))
    return ranges[shard_id]


# this routine stages data for each instance
def stage_instance_data(
        stage_comm, instance_comm, instance_node_comm,
        lsize, lrank,
        hdf5file, dataset, target_directory,
        batch_size=-1,
        stage_num_workers=1,
        stage_mode="node",
        full_dataset_per_node=True,
        use_direct_io=False,
        prepare_staging=False, load_hdf5=False, touch=False
):
    # comm parameters
    # ssize = stage_comm.Get_size()
    # srank = stage_comm.Get_rank()
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    # nsize = instance_node_comm.Get_size()
    # nrank = instance_node_comm.Get_rank()
    f = h5py.File(hdf5file, "r")
    ds = f.get(dataset)
    num_files = ds.shape[0]
    # num_files = 1000

    shard_start, shard_end = get_shard_range(num_files, isize, irank, cycle_dist=lsize)

    chunk_size = 16
    chunk_start = shard_start
    files_local = []
    while True:
        chunk_end = min(shard_end, chunk_start + chunk_size)
        data = ds[chunk_start:chunk_end]
        for i in range(data.shape[0]):
            if dataset == "labels":
                id_ = "label"
            else:
                id_ = dataset
            outputfile = id_ + "-" + "{:06}".format(chunk_start + i) + ".npy"
            if touch:
                subprocess.run(['touch', str(os.path.join(target_directory, outputfile))])
            else:
                np.save(os.path.join(target_directory, outputfile), data[i])
            files_local.append(outputfile)
        if chunk_end == shard_end:
            break
        chunk_start = chunk_end

    return 0, 0


def stage_data_helper(
        global_comm, num_instances, instance_id, instance_comm,
        local_size, local_rank, pargs, verify=False,
        full_dataset_per_node=True, use_direct_io=False,
        seed=333,
        prepare_staging=False, touch=False
):
    # - Every instance needs all the data, so we need inum replicas.
    # - Every rank irank within an instance can stage data_size / isize of the total data
    # - Since there are num_instances ranks working on the same data, we could shard this among
    # those ranks too
    gsize = global_comm.Get_size()
    grank = global_comm.Get_rank()
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    lsize = local_size
    lrank = local_rank


    load_hdf5 = False

    # create staging filter:
    pargs.data_format = "dali-numpy/hdf5"  # TODO Fix
    if False and (pargs.data_format == "dali-numpy") or (pargs.data_format == 'dali-es'):
        stage_filter_list = ['validation/data-*.npy', 'validation/label-*.npy',
                             'train/data-*.npy', 'train/label-*.npy']
        # print("not hdf5", pargs.data_format)
    elif pargs.data_format == "dali-numpy/hdf5" or True:
        stage_filter_list = ["train.h5/data", "train.h5/labels", "validation.h5/data",
                             "validation.h5/labels"]
        load_hdf5 = True
        # print("hdf5!!")
    elif pargs.data_format == "dali-dummy":
        return
    else:
        raise NotImplementedError(
            f"Error, data-format {pargs.data_format} not implemented for staging"
        )

    # create subdirectory for each instance, just in case if multiple instances see the same directory
    stage_dir = os.path.join(pargs.stage_dir_prefix, f"instance{instance_id}")
    if lrank == 0:
        os.makedirs(stage_dir, exist_ok=True)

    # create the train and validation folders
    if lrank == 0:
        os.makedirs(os.path.join(stage_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(stage_dir, "validation"), exist_ok=True)

    # split the global communicator according to irank: key could be instance_id but we would end up
    stage_comm = global_comm.Split(color=irank, key=instance_id)

    # split the instance by nodes and create a comm with all matching local ranks by node
    instance_node_id = irank // lsize
    instance_node_comm = instance_comm.Split(color=lrank, key=instance_node_id)

    # iterate over staging filters
    file_stats = {}
    for stage_filter in stage_filter_list:
        nvtx.range_push(f"stage {stage_filter}")

        if not prepare_staging and (grank == 0):
            print(f"Staging {stage_filter}", flush=True)
        elif irank == 0:
            print(f"Preparing file lists for {stage_dir} {stage_filter}", flush=True)

        # get directories
        if not load_hdf5:
            stage_source_directory = os.path.join(
                pargs.data_dir_prefix, os.path.dirname(stage_filter)
            )
        else:
            tmp = stage_filter.split("/")
            fname, dataset = tmp[0], tmp[1]
            hdf5_file = os.path.join(pargs.data_dir_prefix, fname)
        stage_target_directory = os.path.join(stage_dir, stage_filter.split(".")[0])

        # create target directory if not exist:
        if local_rank == 0:
            os.makedirs(stage_target_directory, exist_ok=True)

        if not load_hdf5:
            # get file info to everybody
            if grank == 0 and not load_hdf5:
                allfiles = sorted(
                    glob(os.path.join(stage_source_directory, os.path.basename(stage_filter)))
                )
            else:
                allfiles = None

            # shuffle files if requested
            if (grank == 0) and (not full_dataset_per_node) and (seed is not None):
                rng = np.random.default_rng(seed)
                rng.shuffle(allfiles)

            # communicate list of files
            allfiles = global_comm.bcast(allfiles, 0)

        # now stage the data so that each rank in each instance has the relevant data
        stage_start = time.perf_counter()
        total_read, total_write = stage_instance_data(
            stage_comm, instance_comm, instance_node_comm,
            lsize, lrank,
            hdf5_file, dataset, stage_target_directory,
            pargs.stage_batch_size,
            pargs.stage_num_workers,
            pargs.stage_mode,
            full_dataset_per_node,
            use_direct_io,
            prepare_staging, load_hdf5=load_hdf5,
            touch=touch
        )
        stage_stop = time.perf_counter()

        # updating file stats buffer
        file_stats[stage_filter] = 0  # len(allfiles)

        # skip the rest if we want to prep staging only
        if prepare_staging:
            continue

        # unit conversion
        unit_convert_gb = 1. / float(1024 * 1024 * 1024)

        # allreduce:
        total_read = global_comm.allreduce(total_read)
        total_write = global_comm.allreduce(total_write)

        # convert units
        total_read *= unit_convert_gb
        total_write *= unit_convert_gb

        # stage duration:
        stage_duration = stage_stop - stage_start

        # print
        if grank == 0:
            print(
                f"""Staging {stage_filter} done.
                      Total number of files: {file_stats[stage_filter]}.
                      Elapsed time {stage_duration:.2f}s. 
                      Read {total_read:.2f} GB (bandwidth: {total_read / stage_duration:.2f} GB/s).
                      Write {total_write:.2f} GB (bandwidth: {total_write / stage_duration:.2f} GB/s).
                   """
            )

        # verify staging results if requested
        if verify:
            nvtx.range_push(f"stage_verify")
            if local_rank == 0:
                files = glob(os.path.join(stage_target_directory, os.path.basename(stage_filter)))
            else:
                files = []

            if not full_dataset_per_node:
                # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
                files_full = instance_comm.allgather(files)
                files_full = set(itertools.chain(*files_full))
            else:
                files_full = set(files)
            num_files = len(files_full)

            # strip off the directory
            checkfiles1 = sorted([os.path.basename(x) for x in files_full])
            checkfiles2 = sorted([os.path.basename(x) for x in allfiles])

            assert (num_files == file_stats[stage_filter])
            assert (checkfiles1 == checkfiles2)

            if irank == 0:
                print(
                    f"Staged data for {stage_filter}: {num_files}, expected: {file_stats[stage_filter]}",
                    flush=True
                )
            nvtx.range_pop()

        # close range
        nvtx.range_pop()

    return 121266, 15158


def touch_files_in_stage_dir(
    global_comm, instance_comm, instance_id, local_size, local_rank, pargs
):
    # need to touch all of the files
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    lrank = local_rank

    # create subdirectory for each instance, just in case if multiple instances see the same directory
    if pargs.data_staging_method == "instance":
        stage_dir = os.path.join(pargs.stage_dir_prefix, f"instance{instance_id}")
    elif pargs.data_staging_method == "nodes":
        stage_dir = os.path.join(pargs.stage_dir_prefix, f"instance{instance_id}")
        node_num = global_comm.Get_rank() // 4
        stage_dir = os.path.join(stage_dir, str(node_num))
    elif pargs.data_staging_method == "full":
        stage_dir = pargs.stage_dir_prefix
    else:
        raise ValueError(f"invalid data staging method: {pargs.data_staging_method}")

    if lrank == 0:
        os.makedirs(stage_dir, exist_ok=True)

    # create the train and validation folders
    train_dir = os.path.join(stage_dir, "train")
    val_dir = os.path.join(stage_dir, "validation")
    if lrank == 0:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

    data_filter = '*data-*.npy'
    label_filter = '*label-*.npy'

    train_data_files = sorted(glob.glob(os.path.join(train_dir, data_filter)))
    train_label_files = sorted(glob.glob(os.path.join(train_dir, label_filter)))

    val_data_files = sorted(glob.glob(os.path.join(train_dir, data_filter)))
    val_label_files = sorted(glob.glob(os.path.join(train_dir, label_filter)))

    # NOTE: THIS CREATES FILES FOR EACH INSTANCE!!

    num_train, num_val = 121266, 15158
    train_shard_sz = num_train // isize
    train_shards = [train_shard_sz * i for i in range(isize)] + [num_train, ]
    val_shard_sz = num_val // isize
    val_shards = [val_shard_sz * i for i in range(isize)] + [num_val, ]
    train_slice = slice(train_shards[irank], train_shards[irank + 1])
    val_slice = slice(val_shards[irank], val_shards[irank + 1])

    itrain_files = train_data_files[train_slice]
    itrain_labels = train_label_files[train_slice]
    for files in itrain_files:
        base_name = os.path.basename(files)
        f = os.path.join(train_dir, base_name)
        subprocess.run(['touch', str(f)])
    for files in itrain_labels:
        base_name = os.path.basename(files)
        f = os.path.join(train_dir, base_name)
        subprocess.run(['touch', str(f)])

    ival_files = val_data_files[val_slice]
    ival_labels = val_label_files[val_slice]
    for files in ival_files:
        base_name = os.path.basename(files)
        f = os.path.join(val_dir, base_name)
        subprocess.run(['touch', str(f)])
    for files in ival_labels:
        base_name = os.path.basename(files)
        f = os.path.join(val_dir, base_name)
        subprocess.run(['touch', str(f)])


def stage_to_NVMe_node_folders_h5(
        global_comm, num_instances, instance_id, instance_comm,
        local_size, local_rank, pargs, verify=False,
        full_dataset_per_node=True, use_direct_io=False,
        seed=333, prepare_staging=False,
        number_workers=6, touch=False
):
    # NOTE: this will use the global comm exclusivly
    # only stage the shard of the data which will go on that node
    # TODO: tell DALI that this data is already staged (use dali-numpy?)
    # each instance gets a full dataset, so we need inum replicas.
    #   REMINDER: data is already shuffled in the file
    #   0. create folder for each node in the NVMe dir -> instance_num/instance_node/(train/val)
    #   1. get full length
    #   2. get number of items per rank in instance

    # - Every rank irank within an instance can stage data_size / isize of the total data
    # - Since there are num_instances ranks working on the same data, we could shard this among
    # those ranks too
    gsize = global_comm.Get_size()
    grank = global_comm.Get_rank()
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    lsize = local_size  # get the number of GPUs on each node
    lrank = local_rank  # get the node-local rank

    # create staging filter:
    if pargs.data_format.endswith("hdf5"):
        stage_filter_list = ["train.h5/data", "train.h5/labels", "validation.h5/data",
                             "validation.h5/labels"]
        # print("hdf5!!")
    elif pargs.data_format == "dali-dummy":
        return
    else:
        raise NotImplementedError(
            f"Error, data-format {pargs.data_format} not implemented for h5 staging"
        )

    stage_dir = os.path.join(pargs.stage_dir_prefix, f"instance{instance_id}")
    node_num = grank // 4
    stage_dir = os.path.join(stage_dir, str(node_num))

    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(os.path.join(stage_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(stage_dir, "validation"), exist_ok=True)

    # print(f"stage dir {stage_dir}")
    # stage_dir -> /NVMe_folder/instance_num/node_number/

    stage_comm = global_comm.Split(color=irank, key=instance_id)

    # iterate over staging filters
    file_stats = {}
    for stage_filter in stage_filter_list:
        nvtx.range_push(f"stage {stage_filter}")

        # if not prepare_staging and (grank == 0):
        #     print(f"Staging {stage_filter}", flush=True)
        # elif grank == 0:  # this should run for the single h5
        #     print(f"Preparing file lists for {stage_filter}", flush=True)

        # get directories
        tmp = stage_filter.split("/")  # split off the data/lable at the end of the stage_filer
        fname, dataset = tmp[0], tmp[1]  # h5 file, (data/label)
        hdf5_file = os.path.join(pargs.data_dir_prefix, fname)
        stage_target_directory = os.path.join(stage_dir, stage_filter.split(".")[0])

        # now stage the data so that each rank in each instance has the relevant data
        stage_start = time.perf_counter()
        # print(f"stage_target_directory: {stage_target_directory}")
        total_read, total_write = stage_instance_data_nvme(
            stage_comm, global_comm, instance_comm, hdf5_file, dataset, stage_target_directory, touch=touch
        )
        stage_stop = time.perf_counter()

        # updating file stats buffer
        file_stats[stage_filter] = 0

        # skip the rest if we want to prep staging only
        if prepare_staging:
            continue

        # unit conversion
        unit_convert_gb = 1. / float(1024 * 1024 * 1024)

        # allreduce:
        total_read = global_comm.allreduce(total_read)
        total_write = global_comm.allreduce(total_write)

        # convert units
        total_read *= unit_convert_gb
        total_write *= unit_convert_gb

        # stage duration:
        stage_duration = stage_stop - stage_start

        # print
        if grank == 0:
            print(
                f"""Staging {stage_filter} done.
                      Total number of files: {file_stats[stage_filter]}.
                      Elapsed time {stage_duration:.2f}s. 
                      Read {total_read:.2f} GB (bandwidth: {total_read / stage_duration:.2f} GB/s).
                      Write {total_write:.2f} GB (bandwidth: {total_write / stage_duration:.2f} GB/s).
                   """
            )

        # verify staging results if requested
        if verify:
            nvtx.range_push(f"stage_verify")
            if local_rank == 0:
                files = glob(os.path.join(stage_target_directory, os.path.basename(stage_filter)))
            else:
                files = []

            if not full_dataset_per_node:
                # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
                files_full = instance_comm.allgather(files)
                files_full = set(itertools.chain(*files_full))
            else:
                files_full = set(files)
            num_files = len(files_full)

            # strip off the directory
            checkfiles1 = sorted([os.path.basename(x) for x in files_full])
            checkfiles2 = sorted([os.path.basename(x) for x in allfiles])

            assert (num_files == file_stats[stage_filter])
            assert (checkfiles1 == checkfiles2)

            if irank == 0:
                print(
                    f"Staged data for {stage_filter}: {num_files}, expected: {file_stats[stage_filter]}",
                    flush=True
                )
            nvtx.range_pop()

        # close range
        nvtx.range_pop()

    return 121266, 15158


def stage_instance_data_nvme(
        stage_comm, global_comm, instance_comm, hdf5file, dataset, target_directory, touch=False
):
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()

    f = h5py.File(hdf5file, "r")
    ds = f.get(dataset)
    num_files = ds.shape[0]
    # num_files = 100

    # get shard range ========================================
    # number of shards is the number of ranks in an instance
    num_shards = isize
    num_files_per_shard = num_files // num_shards
    num_files_remainder = num_files % num_shards

    shard_start = [0]
    for i in range(1, num_shards):
        # ensure that there is an even number of files
        if i - 1 < num_files_remainder:
            this_shard_start = shard_start[-1] + (num_files_per_shard + 1)
        else:
            this_shard_start = shard_start[-1] + num_files_per_shard
        shard_start.append(this_shard_start)
    shard_start.append(num_files)

    ranges = []
    for i in range(num_shards):
        ranges.append((shard_start[i], shard_start[i + 1]))
    shard_start, shard_end = ranges[irank]

    chunk_size = 32
    chunk_start = shard_start
    files_local = []

    while True:
        chunk_end = min(shard_end, chunk_start + chunk_size)
        data = ds[chunk_start:chunk_end]
        for i in range(data.shape[0]):
            if dataset == "labels":
                id_ = "label"
            else:
                id_ = dataset
            outputfile = id_ + "-" + "{:06}".format(chunk_start + i) + ".npy"
            if touch:
                subprocess.run(['touch', str(os.path.join(target_directory, outputfile))])
            else:
                np.save(os.path.join(target_directory, outputfile), data[i])
            files_local.append(outputfile)
        if chunk_end == shard_end:
            break
        chunk_start = chunk_end

    return 0, 0


def stage_to_NVMe_all_shared_h5(
        global_comm, num_instances, instance_id, instance_comm,
        local_size, local_rank, pargs, verify=False,
        full_dataset_per_node=True, use_direct_io=False,
        seed=333, prepare_staging=False,
        number_workers=6, touch=False,
):
    # NOTE: this will use the global comm exclusivly
    # only stage the shard of the data which will go on that node
    # each instance gets a full dataset, so we need inum replicas.
    #   REMINDER: data is already shuffled in the file
    #   0. create folder for each node in the NVMe dir -> instance_num/instance_node/(train/val)
    #   1. get full length
    #   2. get number of items per rank in instance

    # - Every rank irank within an instance can stage data_size / isize of the total data
    # - Since there are num_instances ranks working on the same data, we could shard this among
    # those ranks too
    gsize = global_comm.Get_size()
    grank = global_comm.Get_rank()
    lsize = local_size  # get the number of GPUs on each node
    lrank = local_rank  # get the node-local rank
    # print(f"Start staging, gsize {gsize} grank {grank} lsize {lsize} lrank {lrank}")

    if pargs.data_format.endswith("hdf5"):
        stage_filter_list = ["train.h5/data", "train.h5/labels", "validation.h5/data",
                             "validation.h5/labels"]
    elif pargs.data_format == "dali-dummy":
        return
    else:
        raise NotImplementedError(
            f"Error, data-format {pargs.data_format} not implemented for h5 staging"
        )

    stage_dir = pargs.stage_dir_prefix

    if lrank == 0:
        os.makedirs(os.path.join(stage_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(stage_dir, "validation"), exist_ok=True)

    # print(f"stage dir {stage_dir}")
    # stage_dir -> /NVMe_folder/

    # iterate over staging filters
    file_stats = {}
    for stage_filter in stage_filter_list:
        nvtx.range_push(f"stage {stage_filter}")

        if not prepare_staging and (grank == 0):
            print(f"Staging {stage_filter}", flush=True)
        elif grank == 0:  # this should run for the single h5
            print(f"Preparing file lists for {stage_filter}", flush=True)


        tmp = stage_filter.split("/")  # split off the data/lable at the end of the stage_filer
        fname, dataset = tmp[0], tmp[1]  # h5 file, (data/label)
        hdf5_file = os.path.join(pargs.data_dir_prefix, fname)
        stage_target_directory = os.path.join(stage_dir, stage_filter.split(".")[0])

        # now stage the data so that each rank in each instance has the relevant data
        stage_start = time.perf_counter()
        # print(f"stage_target_directory: {stage_target_directory} touch: {touch}")
        total_read, total_write = stage_instance_data_nvme_all_shared(
            global_comm, hdf5_file, dataset, stage_target_directory, stage_dir, touch=touch
        )
        stage_stop = time.perf_counter()

        # updating file stats buffer
        file_stats[stage_filter] = 0  # len(allfiles)

        # skip the rest if we want to prep staging only
        if prepare_staging:
            continue

        # unit conversion
        unit_convert_gb = 1. / float(1024 * 1024 * 1024)

        # allreduce:
        total_read = global_comm.allreduce(total_read)
        total_write = global_comm.allreduce(total_write)

        # convert units
        total_read *= unit_convert_gb
        total_write *= unit_convert_gb

        # stage duration:
        stage_duration = stage_stop - stage_start

        # print
        if grank == 0:
            print(
                f"""Staging {stage_filter} done.
                      Total number of files: {file_stats[stage_filter]}.
                      Elapsed time {stage_duration:.2f}s. 
                      Read {total_read:.2f} GB (bandwidth: {total_read / stage_duration:.2f} GB/s).
                      Write {total_write:.2f} GB (bandwidth: {total_write / stage_duration:.2f} GB/s).
                   """
            )

        # verify staging results if requested
        if verify:
            nvtx.range_push(f"stage_verify")
            if local_rank == 0:
                files = glob(os.path.join(stage_target_directory, os.path.basename(stage_filter)))
            else:
                files = []

            if not full_dataset_per_node:
                # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
                files_full = instance_comm.allgather(files)
                files_full = set(itertools.chain(*files_full))
            else:
                files_full = set(files)
            num_files = len(files_full)

            # strip off the directory
            checkfiles1 = sorted([os.path.basename(x) for x in files_full])
            checkfiles2 = sorted([os.path.basename(x) for x in allfiles])

            assert (num_files == file_stats[stage_filter])
            assert (checkfiles1 == checkfiles2)

            if irank == 0:
                print(
                    f"Staged data for {stage_filter}: {num_files}, expected: {file_stats[stage_filter]}",
                    flush=True
                )
            nvtx.range_pop()

        # close range
        nvtx.range_pop()

    global_comm.Barrier()

    return 121266, 15158


def stage_instance_data_nvme_all_shared(
        global_comm, hdf5file, dataset, target_directory, stage_dir, touch=False
):
    gsize = global_comm.Get_size()
    grank = global_comm.Get_rank()
    f = h5py.File(hdf5file, "r")
    ds = f.get(dataset)
    num_files = ds.shape[0]
    # num_files = 100

    # get shard range ========================================
    # number of shards is the number of ranks in an instance
    num_shards = gsize
    num_files_per_shard = num_files // num_shards
    num_files_remainder = num_files % num_shards

    shard_start = [0]
    for i in range(1, num_shards):
        # ensure that there is an even number of files
        if i - 1 < num_files_remainder:
            this_shard_start = shard_start[-1] + (num_files_per_shard + 1)
        else:
            this_shard_start = shard_start[-1] + num_files_per_shard
        shard_start.append(this_shard_start)
    shard_start.append(num_files)

    ranges = []
    for i in range(num_shards):
        ranges.append((shard_start[i], shard_start[i + 1]))
    shard_start, shard_end = ranges[grank]

    chunk_size = 16
    chunk_start = shard_start

    while True:
        chunk_end = min(shard_end, chunk_start + chunk_size)
        data = ds[chunk_start:chunk_end]
        for i in range(data.shape[0]):
            if dataset == "labels":
                id_ = "label"
                outputfile = id_ + "-" + "{:06}".format(chunk_start + i) + ".npy"
            else:
                id_ = dataset
                outputfile = id_ + "-" + "{:06}".format(chunk_start + i) + ".npy"

            if not touch:
                np.save(os.path.join(target_directory, outputfile), data[i])
            else:
                subprocess.run(['touch', str(os.path.join(target_directory, outputfile))])
        if chunk_end == shard_end:
            break
        chunk_start = chunk_end

    return 0, 0


def stage_to_NVMe_instance_rank_folders_h5(
        global_comm, num_instances, instance_id, instance_comm,
        local_size, local_rank, pargs, verify=False,
        full_dataset_per_node=True, use_direct_io=False,
        seed=333, prepare_staging=False,
        number_workers=6, touch=False
):
    # NOTE: this will use the global comm exclusivly
    # only stage the shard of the data which will go on that node
    # TODO: tell DALI that this data is already staged (use dali-numpy?)
    # each instance gets a full dataset, so we need inum replicas.
    #   REMINDER: data is already shuffled in the file
    #   0. create folder for each node in the NVMe dir -> instance_num/instance_node/(train/val)
    #   1. get full length
    #   2. get number of items per rank in instance

    # - Every rank irank within an instance can stage data_size / isize of the total data
    # - Since there are num_instances ranks working on the same data, we could shard this among
    # those ranks too
    # gsize = global_comm.Get_size()
    grank = global_comm.Get_rank()
    # isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    # lsize = local_size  # get the number of GPUs on each node
    # lrank = local_rank  # get the node-local rank
    # print(f"Start staging, gsize {gsize} grank {grank} isize {isize} irank {irank} lsize {lsize} "
    #       f"lrank {lrank}")

    if pargs.data_format.endswith("hdf5"):
        stage_filter_list = ["train.h5/data", "train.h5/labels", "validation.h5/data",
                             "validation.h5/labels"]
        # print("hdf5!!")
    elif pargs.data_format == "dali-dummy":
        return
    else:
        raise NotImplementedError(
            f"Error, data-format {pargs.data_format} not implemented for h5 staging"
        )

    # create subdirectory for each instance, just in case if multiple instances see the same directory

    node_number = grank // 4  # 4 gpus per node
    stage_dir = os.path.join(pargs.stage_dir_prefix, str(irank))
    os.makedirs(stage_dir, exist_ok=True)
    # print(f"{grank} {lrank} {node_number} stage dir: {stage_dir}")

    # if lrank == 0:  # should be fine to try to make it from every rank
    os.makedirs(os.path.join(stage_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(stage_dir, "validation"), exist_ok=True)

    print(f"stage dir {stage_dir}")

    # stage_dir -> /NVMe_folder/instance_rank

    # iterate over staging filters
    file_stats = {}
    for stage_filter in stage_filter_list:
        nvtx.range_push(f"stage {stage_filter}")

        if not prepare_staging and (grank == 0):
            print(f"Staging {stage_filter}", flush=True)
        elif grank == 0:  # this should run for the single h5
            print(f"Preparing file lists for {stage_filter}", flush=True)

        tmp = stage_filter.split("/")  # split off the data/lable at the end of the stage_filer
        fname, dataset = tmp[0], tmp[1]  # h5 file, (data/label)
        hdf5_file = os.path.join(pargs.data_dir_prefix, fname)
        stage_target_directory = os.path.join(stage_dir, stage_filter.split(".")[0])

        # now stage the data so that each rank in each instance has the relevant data
        stage_start = time.perf_counter()
        print(f"stage_target_directory: {stage_target_directory}")
        total_read, total_write = stage_instance_data_nvme_instance_ranks(
            global_comm, instance_comm, hdf5_file, dataset, stage_target_directory, instance_id, touch=touch
        )
        stage_stop = time.perf_counter()

        # updating file stats buffer
        file_stats[stage_filter] = 0  # len(allfiles)

        # skip the rest if we want to prep staging only
        if prepare_staging:
            continue

        # unit conversion
        unit_convert_gb = 1. / float(1024 * 1024 * 1024)

        # allreduce:
        total_read = global_comm.allreduce(total_read)
        total_write = global_comm.allreduce(total_write)

        # convert units
        total_read *= unit_convert_gb
        total_write *= unit_convert_gb

        # stage duration:
        stage_duration = stage_stop - stage_start

        # print
        if grank == 0:
            print(
                f"""Staging {stage_filter} done.
                      Total number of files: {file_stats[stage_filter]}.
                      Elapsed time {stage_duration:.2f}s. 
                      Read {total_read:.2f} GB (bandwidth: {total_read / stage_duration:.2f} GB/s).
                      Write {total_write:.2f} GB (bandwidth: {total_write / stage_duration:.2f} GB/s).
                   """
            )

        # verify staging results if requested
        if verify:
            nvtx.range_push(f"stage_verify")
            if local_rank == 0:
                files = glob(os.path.join(stage_target_directory, os.path.basename(stage_filter)))
            else:
                files = []

            if not full_dataset_per_node:
                # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
                files_full = instance_comm.allgather(files)
                files_full = set(itertools.chain(*files_full))
            else:
                files_full = set(files)
            num_files = len(files_full)

            # strip off the directory
            checkfiles1 = sorted([os.path.basename(x) for x in files_full])
            checkfiles2 = sorted([os.path.basename(x) for x in allfiles])

            assert (num_files == file_stats[stage_filter])
            assert (checkfiles1 == checkfiles2)

            if irank == 0:
                print(
                    f"Staged data for {stage_filter}: {num_files}, expected: {file_stats[stage_filter]}",
                    flush=True
                )
            nvtx.range_pop()

        # close range
        nvtx.range_pop()

    return 121266, 15158


def stage_instance_data_nvme_instance_ranks(
        global_comm, instance_comm, hdf5file, dataset, target_directory, instance_id, touch=False
):
    gsize = global_comm.Get_size()
    # grank = global_comm.Get_rank()
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    # lsize = 4  # local_size  # get the number of GPUs on each node
    # lrank = grank // 4  # get the node-local rank
    f = h5py.File(hdf5file, "r")
    ds = f.get(dataset)
    num_files = ds.shape[0]
    # num_files = 100

    # get shard range ========================================
    # number of shards is the number of ranks in an instance
    num_shards = isize
    num_files_per_shard = num_files // num_shards
    num_files_remainder = num_files % num_shards
    # this is the sharding within each instance

    shard_start = [0]
    for i in range(1, num_shards):
        # ensure that there is an even number of files
        if i - 1 < num_files_remainder:
            this_shard_start = shard_start[-1] + (num_files_per_shard + 1)
        else:
            this_shard_start = shard_start[-1] + num_files_per_shard
        shard_start.append(this_shard_start)
    shard_start.append(num_files)

    ranges = []
    for i in range(num_shards):
        ranges.append((shard_start[i], shard_start[i + 1]))
    shard_start, shard_end = ranges[irank]
    # these ranges need to be split into the number of instances:
    #   st ----i0-------i1-------i2-------i3--------i4------i5----- sp
    num_instances = gsize // isize
    stepsize = (shard_end - shard_start) // num_instances
    instance_shards = [shard_start + i * stepsize for i in range(num_instances)]
    instance_shards.append(shard_end)
    shard_start, shard_end = instance_shards[instance_id], instance_shards[instance_id + 1]
    # ========================================================
    chunk_size = 16
    chunk_start = shard_start
    files_local = []

    while True:
        chunk_end = min(shard_end, chunk_start + chunk_size)
        data = ds[chunk_start:chunk_end]
        for i in range(data.shape[0]):
            if dataset == "labels":
                id_ = "label"
            else:
                id_ = dataset
            outputfile = id_ + "-" + "{:06}".format(chunk_start + i) + ".npy"
            if touch:
                subprocess.run(['touch', str(os.path.join(target_directory, outputfile))])
            else:
                np.save(os.path.join(target_directory, outputfile), data[i])

            files_local.append(outputfile)
        if chunk_end == shard_end:
            break
        chunk_start = chunk_end

    return 0, 0
