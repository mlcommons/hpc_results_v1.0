import io
import os
import time

import h5py
from mpi4py import MPI
import numpy as np
# import tarfile


def load_numpy(tar_file, tar_member):
    with tar_file.extractfile(tar_member) as f:
        return np.load(io.BytesIO(f.read()))


np.random.seed(42)

print("this is the h5py file we are using:", h5py.__file__)

# # root_file="/p/scratch/jb_benchmark/cosmoUniverse_2019_05_4parE_tf_v2_numpy.tar"
# root_file="/hkfs/work/workspace/scratch/qv2382-mlperf-cosmo-data/" \
#           "/p/ime-scratch/fs/jb_benchmark/cosmoUniverse_2019_05_4parE_tf_v2_numpy.tar"
# # Use either "train" or "validation"
# data_subset = "train"
# with tarfile.open(root_file, 'r') as tar_f:
#     start_time = time.perf_counter()
#     files = [
#         n
#         for n in tar_f.getmembers()
#         if n.name.startswith(
#                 f'cosmoUniverse_2019_05_4parE_tf_v2_numpy/{data_subset}')
#     ]


if __name__ == "__main__":
    root_dir = "/hkfs/work/workspace/scratch/qv2382-mlperf-cosmo-data/new_npy_files"
    for tv in ["train", "validation"]:
        target_dir = os.path.join(root_dir, tv)  # "validation"
        target_files = os.listdir(target_dir)
        print(tv, target_dir)

        data_files = list(filter(lambda x: x.endswith("data.npy"), target_files))
        data_files.sort()
        data_files = np.array(data_files)
        label_files = list(filter(lambda x: x.endswith("label.npy"), target_files))
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

        first_data = np.load(os.path.join(target_dir, data_files[0]))
        data_shape = first_data.shape
        data_dtype = first_data.dtype
        first_label = np.load(os.path.join(target_dir, label_files[0]))
        label_shape = first_label.shape
        label_dtype = first_label.dtype
        all_data_shape = (len(data_files_filtered), *data_shape)
        all_label_shape = (len(data_files_filtered), *label_shape)


        def write_to_h5_file(data_files, label_files, tfname, start_entry, all_data_shape, all_label_shape,
                             tell_progress=200):
            # if MPI.COMM_WORLD.rank == 0 and os.path.isfile(tfname):
            #     print("file exists")
            #     exit(1)
            #     os.remove(tfname)
            MPI.COMM_WORLD.Barrier()
            print("creating file")
            fi = h5py.File(tfname, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            print("creating dset")
            dset = fi.create_dataset('data', all_data_shape, dtype=data_dtype)
            print("creating lset")
            lset = fi.create_dataset('label', all_label_shape, dtype=label_dtype)

            startt = time.time()
            for ii, (f, l) in enumerate(zip(data_files, label_files)):
                try:
                    data = np.load(f)
                    label = np.load(l)
                except:
                    print(f"ERROR: remove file {f}, {l}")
                dset[start_entry + ii] = data
                lset[start_entry + ii] = label
                now = time.time()
                # time_remaining = len(data_files) * (now - startt) / (ii + 1)
                # if ii % tell_progress == 0:
                #     print(ii, time_remaining / 60, f, l)
            print("closing file")
            fi.close()
            print("SUCCESS, end of write function")


        my_data_files = [os.path.join(target_dir, f) for f in data_files_shards[MPI.COMM_WORLD.rank]]
        my_label_files = [os.path.join(target_dir, f) for f in label_files_shards[MPI.COMM_WORLD.rank]]
        start_entry = start_entries[MPI.COMM_WORLD.rank]

        # all_data_files = np.concatenate(MPI.COMM_WORLD.allgather(
        #     [m.name for m in my_data_files]))
        # print(my_data_files[:2])
        all_data_files = np.concatenate(MPI.COMM_WORLD.allgather(my_data_files))

        print(len(np.unique(all_data_files)), "unique files, and ", len(all_data_files), " total files.")
        if len(np.unique(all_data_files)) != len(all_data_files):
            print("There is an error with the file distribution")

        hdf5file = f"/hkfs/work/workspace/scratch/qv2382-mlperf-cosmo-data/h5_files/{tv}.h5"
        files_file = f"/hkfs/work/workspace/scratch/qv2382-mlperf-cosmo-data/h5_files/{tv}.h5.files"
        if MPI.COMM_WORLD.rank == 0:
            with open(files_file, "w") as g:
                g.write("\n".join(all_data_files) + '\n')

        write_start_time = time.perf_counter()
        write_to_h5_file([f for f in my_data_files],
                         [f for f in my_label_files], hdf5file, start_entry, all_data_shape, all_label_shape)
        print('finished on', MPI.COMM_WORLD.rank, 'after',
              time.perf_counter() - write_start_time, 'seconds')
