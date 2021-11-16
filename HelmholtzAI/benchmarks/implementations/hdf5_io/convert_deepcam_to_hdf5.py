import h5py
import tarfile
import os
import numpy as np
import time
# import tqdm
from mpi4py import MPI

np.random.seed(42)

print("this is the h5py file we are using:", h5py.__file__)


def write_to_h5_file(
    data_files, label_files, tfname, start_entry, all_data_shape, all_label_shape, tell_progress=10
):
    if MPI.COMM_WORLD.rank == 0 and os.path.isfile(tfname):
        print("file exists")
        exit(1)
        os.remove(tfname)
    # MPI.COMM_WORLD.Barrier()
    print("creating file")
    fi = h5py.File(tfname, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    print("creating dset")
    dset = fi.create_dataset('data', all_data_shape, dtype='f')
    print("creating lset")
    lset = fi.create_dataset('labels', all_label_shape, dtype='f')

    startt = time.time()
    for ii, (f, l) in enumerate(zip(data_files, label_files)):
        data = np.load(f)
        label = np.load(l)
        dset[start_entry + ii] = data
        lset[start_entry + ii] = label
        now = time.time()
        time_remaining = len(data_files) * (now - startt) / (ii + 1)
        if ii % tell_progress == 0:
            print(ii, time_remaining / 60, f, l)
    fi.close()


if __name__ == "__main__":
    root_dir = "/hkfs/home/dataset/datasets/deepcam_npy"
    for tv in ["train", "validation"]:
        print(tv)
        target_dir = os.path.join(root_dir, tv)  # "validation"
        target_files = os.listdir(target_dir)

        data_files = list(filter(lambda x: x.endswith("npy") and x.startswith("data"), target_files))
        data_files.sort()
        data_files = np.array(data_files)
        label_files = list(filter(lambda x: x.endswith("npy") and x.startswith("label"), target_files))
        label_files.sort()
        label_files = np.array(label_files)
        perm = np.random.permutation(len(label_files))
        data_files = data_files[perm]
        label_files = label_files[perm]

        no_shards = MPI.COMM_WORLD.size
        data_files_filtered = data_files[:]
        label_files_filtered = label_files[:]
        data_files_shards = []
        label_files_shards = []
        for i in range(no_shards):
            shard_size = int(np.ceil(len(data_files_filtered) / no_shards))
            start = i * shard_size
            end = min((i + 1) * shard_size, len(data_files_filtered))
            data_files_shards.append(data_files_filtered[start:end])
            label_files_shards.append(label_files_filtered[start:end])

        start_entries = np.cumsum([len(x) for x in data_files_shards])
        start_entries = ([0] + list(start_entries))[:-1]

        data_shape = np.load(os.path.join(target_dir, data_files[0])).shape
        label_shape = np.load(os.path.join(target_dir, label_files[0])).shape
        all_data_shape = (len(data_files_filtered), *data_shape)
        all_label_shape = (len(data_files_filtered), *label_shape)

        rank = MPI.COMM_WORLD.rank
        my_data_files = [os.path.join(target_dir, f) for f in data_files_shards[rank]]
        my_label_files = [os.path.join(target_dir, f) for f in label_files_shards[rank]]
        start_entry = start_entries[rank]

        all_data_files = np.concatenate(MPI.COMM_WORLD.allgather(my_data_files))

        print(
            f"{len(np.unique(all_data_files))} unique files and {len(all_data_files)} total files."
        )
        if len(np.unique(all_data_files)) != len(all_data_files):
            print("There is an error with the file distribution")

        hdf5file = f"/hkfs/work/workspace/scratch/qv2382-mlperf_data/hdf5s/{tv}.h5"
        files_file = f"/hkfs/work/workspace/scratch/qv2382-mlperf_data/hdf5s/{tv}.h5.files"
        with open(files_file, "w") as g:
            g.write("\n".join(all_data_files))

        write_to_h5_file(
            [os.path.join(target_dir, f) for f in my_data_files],
            [os.path.join(target_dir, f) for f in my_label_files], hdf5file, start_entry, all_data_shape,
            all_label_shape
        )
