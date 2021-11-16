import atexit
import argparse
import math
import os
import pathlib
from typing import Callable, List, Tuple

import h5py
import numpy as np
import nvidia.dali.fn as dali_fn
import nvidia.dali.math as dali_math
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import nvidia.dali.plugin.mxnet as dali_mxnet
import nvidia.dali.types as dali_types

import data as datam  # "datam" = "data module"
import utils


@utils.ArgumentParser.register_extension("HDF5 Pipeline")
def add_h5_argument_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use_h5",
        action="store_true",
        help="Whether to use HDF5 data.",
    )
    parser.add_argument(
        "--read_chunk_size",
        type=int,
        default=32,
        help=(
            "How many rows at once to read from the HDF5 data when "
            "prestaging and not preshuffling."
        ),
    )


def stage_files(
        dist_desc: utils.DistributedEnvDesc,
        data_dir: pathlib.Path,
        output_dir: pathlib.Path,
        data_filenames: List[str],
        label_filenames: List[str],
        shard_mult: int,
        preshuffle_permutation: np.ndarray,
        read_chunk_size: int,
) -> Tuple[List[str], List[str], Callable]:
    number_of_nodes = dist_desc.size // dist_desc.local_size // shard_mult
    current_node = dist_desc.rank // dist_desc.local_size // shard_mult
    files_per_node = len(data_filenames) // number_of_nodes
    assert (
        preshuffle_permutation is not None
        or dist_desc.size * read_chunk_size <= len(data_filenames) * shard_mult
    ), '`read_chunk_size` is too high and will cause errors'

    if preshuffle_permutation is not None:
        read_chunk_size = 1
        all_indices = np.arange(len(data_filenames))
        all_indices = all_indices[preshuffle_permutation]
        indices_per_node = files_per_node
    else:
        all_indices = np.arange(0, len(data_filenames), read_chunk_size)
        indices_per_node = len(all_indices) // number_of_nodes

    per_node_indices = all_indices[
        current_node * indices_per_node:(current_node+1) * indices_per_node]
    per_node_data_filenames = data_filenames[
        current_node * files_per_node:(current_node+1) * files_per_node]
    per_node_label_filenames = label_filenames[
        current_node * files_per_node:(current_node+1) * files_per_node]

    os.makedirs(output_dir, exist_ok=True)
    # print("started staging files")
    h5_path = data_dir / (data_dir.name + '.h5')

    with h5py.File(h5_path, 'r') as h5_file:
        data_dset = h5_file['data']
        label_dset = h5_file['label']
        copied_files = 0
        per_shard_data_filenames = per_node_data_filenames[
            dist_desc.local_rank::dist_desc.local_size]
        per_shard_label_filenames = per_node_label_filenames[
            dist_desc.local_rank::dist_desc.local_size]
        for (i, f) in zip(
                per_node_indices[
                    dist_desc.local_rank::dist_desc.local_size],
                range(
                    0,
                    len(per_shard_data_filenames),
                    read_chunk_size,
                ),
        ):
            next_i = i + read_chunk_size
            next_f = f + read_chunk_size

            np_datas = data_dset[i:next_i]
            for (np_data, data) in zip(
                    np_datas,
                    per_shard_data_filenames[f:next_f],
            ):
                np.save(output_dir / data, np_data)

            np_labels = label_dset[i:next_i]
            for (np_label, label) in zip(
                    np_labels,
                    per_shard_label_filenames[f:next_f],
            ):
                np.save(output_dir / label, np_label)

            copied_files += len(np_datas)

    # print(f"Node {current_node}, process {dist_desc.local_rank}, "
    #       f"dataset contains {len(data_filenames)} samples, "
    #       f"per node {len(per_node_data_filenames)}, copied {copied_files}")

    dist_desc.comm.Barrier()
    return per_node_data_filenames, per_node_label_filenames


class ShardedH5Iterator:
    ___slots___ = [
        "NUM_OUTPUTS",
        "batch_size",
        "shard_indices",
        "preshuffle",
        "curr_index",
        "h5_file",
        "data_dset",
        "label_dset",
    ]

    NUM_OUTPUTS = 2

    def __init__(
            self,
            h5_path,
            num_samples,
            batch_size,
            shard_type,
            num_shards,
            shard_id,
            dist_desc,
            shard_mult,
            preshuffle_permutation,
    ):
        self.batch_size = batch_size

        self.preshuffle = preshuffle_permutation is not None

        if shard_type == 'local':
            number_of_nodes = \
                dist_desc.size // dist_desc.local_size // shard_mult
            current_node = dist_desc.rank // dist_desc.local_size // shard_mult

            if self.preshuffle:
                all_indices = np.arange(0, num_samples)
                all_indices = all_indices[preshuffle_permutation]

                indices_per_node = num_samples // number_of_nodes
            else:
                all_indices = np.arange(
                    0,
                    num_samples,
                    self.batch_size,
                )

                indices_per_node = len(all_indices) // number_of_nodes

            next_node = current_node + 1
            per_node_indices = all_indices[
                current_node * indices_per_node:next_node * indices_per_node]
            self.shard_indices = per_node_indices[
                dist_desc.local_rank::dist_desc.local_size]
        elif shard_type == 'global':
            if self.preshuffle:
                all_indices = np.arange(0, num_samples)
                all_indices = all_indices[preshuffle_permutation]
            else:
                all_indices = np.arange(
                    0,
                    num_samples,
                    self.batch_size,
                )

            indices_per_shard = len(all_indices) // num_shards

            next_shard = shard_id + 1
            self.shard_indices = all_indices[
                shard_id * indices_per_shard:next_shard * indices_per_shard]
        else:
            raise NotImplementedError

        atexit.register(self.clean_up)
        self.h5_file = h5py.File(h5_path, 'r')
        self.data_dset = self.h5_file['data']
        self.label_dset = self.h5_file['label']

    # The reason we choose this more weird setup is because DALI handles
    # generator functions with some magic. We try to avoid their magic.
    def __iter__(self):
        self.curr_index = 0
        return self

    def __next__(self):
        if self.curr_index >= len(self.shard_indices):
            raise StopIteration

        if self.preshuffle:
            next_index = self.curr_index + self.batch_size
            indices = self.shard_indices[self.curr_index:next_index]
            data_batch = self.data_dset[indices]
            label_batch = self.label_dset[indices]

            self.curr_index = next_index
        else:
            i = self.shard_indices[self.curr_index]
            data_batch = self.data_dset[i:i + self.batch_size]
            label_batch = self.label_dset[i:i + self.batch_size]

            self.curr_index += 1

        return (data_batch, label_batch)

    def clean_up(self):
        try:
            self.h5_file.close()
        except Exception:
            pass


class H5CosmoDataset(datam.CosmoDataset):
    def training_dataset(
            self,
            batch_size: int,
            shard: str = "global",
            shuffle: bool = False,
            preshuffle: bool = False,
            n_samples: int = -1,
            shard_mult: int = 1,
            prestage: bool = False,
            read_chunk_size: int = 32,
    ) -> Tuple[dali_mxnet.DALIGenericIterator, int, int]:
        data_path = self.root_dir / "train"

        pipeline_builder, samples = self._construct_pipeline(
            data_path,
            batch_size,
            n_samples=n_samples,
            shard=shard,
            shuffle=shuffle,
            prestage=(shard == "local") and prestage,
            preshuffle=preshuffle,
            shard_mult=shard_mult,
            read_chunk_size=read_chunk_size,
        )

        # assert samples % self.dist.size == 0, \
        #     f"Cannot divide {samples} items into {self.dist.size} workers"

        iter_count = samples // self.dist.size // batch_size

        def iterator_builder():
            pipeline = pipeline_builder()
            iterator = dali_mxnet.DALIGluonIterator(
                pipeline,
                reader_name='data_reader' if prestage else None,
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )
            return iterator

        return iterator_builder, iter_count, samples

    def validation_dataset(
            self,
            batch_size: int,
            shard: bool = True,
            n_samples: int = -1,
            read_chunk_size: int = 32,
    ) -> Tuple[dali_mxnet.DALIGenericIterator, int, int]:
        data_path = self.root_dir / "validation"

        pipeline_builder, samples = self._construct_pipeline(
            data_path,
            batch_size,
            n_samples=n_samples,
            shard="global",
            prestage=False,
            read_chunk_size=read_chunk_size,
        )
        # assert samples % self.dist.size == 0 or not shard, \
        #     f"Cannot divide {samples} items into {self.dist.size} workers"

        iter_count = samples // (self.dist.size if shard else 1) // batch_size

        def iterator_builder():
            pipeline = pipeline_builder()
            iterator = dali_mxnet.DALIGluonIterator(
                pipeline,
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )
            return iterator

        return iterator_builder, iter_count, samples

    def _construct_pipeline(
            self,
            data_dir: pathlib.Path,
            batch_size: int,
            n_samples: int = -1,
            prestage: bool = True,
            shard: datam.ShardType = "none",
            shuffle: bool = False,
            preshuffle: bool = False,
            shard_mult: int = 1,
            read_chunk_size: int = 32,
    ) -> Tuple[Pipeline, int]:
        data_filenames = datam._load_file_list(data_dir, "files_data.lst")
        label_filenames = datam._load_file_list(data_dir, "files_label.lst")

        if n_samples > 0:
            data_filenames = data_filenames[:n_samples]
            label_filenames = label_filenames[:n_samples]
        n_samples = len(data_filenames) * self.samples_per_file

        if preshuffle:
            preshuffle_permutation = np.ascontiguousarray(
                np.random.permutation(n_samples))
            self.dist.comm.Bcast(preshuffle_permutation, root=0)

            # This only works if DALI does not care about the
            # filename list order.
            # If that is not the case, we could instead shuffle the
            # indices querying the HDF5 file.
            # Finally, since the data is already pre-shuffled,
            # we can also just omit this.
            # Since we now shuffle the indices (resulting in slower
            # staging but safer preshuffling), we comment this out. If
            # we didn't comment this out, preshuffling would have no
            # effect.

            # data_filenames = \
            #     list(np.array(data_filenames)[preshuffle_permutation])
            # label_filenames = \
            #     list(np.array(label_filenames)[preshuffle_permutation])
        else:
            preshuffle_permutation = None

        if shard == "local":
            if shard_mult == 1:
                shard_id = self.dist.local_rank
                num_shards = self.dist.local_size
            else:
                node_in_chunk = self.dist.node % shard_mult
                num_shards = self.dist.local_size * shard_mult
                shard_id = (
                    node_in_chunk
                    * self.dist.local_size + self.dist.local_rank
                )
        elif shard == "global":
            shard_id, num_shards = self.dist.rank, self.dist.size
        else:
            shard_id, num_shards = 0, 1

        def pipeline_builder():
            data_filenames_ = data_filenames
            label_filenames_ = label_filenames
            data_dir_ = data_dir

            if prestage:
                output_path = (
                    pathlib.Path("/staging_area", "dataset")
                    / data_dir.parts[-1]
                )
                data_filenames_, label_filenames_ = stage_files(
                    self.dist,
                    data_dir,
                    output_path,
                    data_filenames,
                    label_filenames,
                    shard_mult,
                    preshuffle_permutation,
                    read_chunk_size,
                )
                data_dir_ = output_path

                return datam.get_dali_pipeline(
                    data_dir_,
                    data_filenames_,
                    label_filenames_,
                    dont_use_mmap=not self.use_mmap,
                    shard_id=shard_id,
                    num_shards=num_shards,
                    apply_log=self.apply_log,
                    batch_size=batch_size,
                    dali_threads=self.threads,
                    device_id=self.dist.local_rank,
                    shuffle=shuffle,
                    data_layout=self.data_layout,
                    sample_shape=self.data_shapes[0],
                    target_shape=self.data_shapes[1],
                    seed=self.seed,
                )
            else:
                h5_path = data_dir / (data_dir.name + '.h5')
                pipe = Pipeline(
                    batch_size=batch_size,
                    num_threads=self.threads,
                    device_id=self.dist.local_rank,
                )
                external_source = ShardedH5Iterator(
                    h5_path,
                    len(data_filenames),
                    batch_size,
                    shard,
                    num_shards,
                    shard_id,
                    self.dist,
                    shard_mult,
                    preshuffle_permutation,
                )

                SAMPLE_SIZE_DATA = 4 * math.prod(self.data_shapes[0])
                # SAMPLE_SIZE_LABEL = 4 * math.prod(self.data_shapes[1])
                with pipe:
                    data, label = dali_fn.external_source(
                        source=external_source,
                        num_outputs=ShardedH5Iterator.NUM_OUTPUTS,
                        cycle='raise',
                    )

                    feature_map = dali_fn.cast(
                        data.gpu(),
                        dtype=dali_types.FLOAT,
                        bytes_per_sample_hint=SAMPLE_SIZE_DATA * batch_size,
                    )
                    if self.apply_log:
                        feature_map = dali_math.log(feature_map + 1.0)
                    else:
                        feature_map = (
                            feature_map
                            / dali_fn.reductions.mean(feature_map)
                        )
                    if self.data_layout == "NCDHW":
                        feature_map = dali_fn.transpose(
                            feature_map,
                            perm=[3, 0, 1, 2],
                        )
                    pipe.set_outputs(feature_map, label.gpu())
                pipe.build()
                return pipe

        return (pipeline_builder, n_samples)


# We could also rewrite this method in `data.py` to pick the dataset
# class depending on the `args`. However, this is more forward-compatible.
def get_rec_iterators(
        args: argparse.Namespace,
        dist_desc: utils.DistributedEnvDesc,
) -> Tuple[Callable, int, int]:
    cosmoflow_dataset = H5CosmoDataset(
        args.data_root_dir,
        dist=dist_desc,
        use_mmap=args.dali_use_mmap,
        apply_log=args.apply_log_transform,
        dali_threads=args.dali_num_threads,
        data_layout=args.data_layout,
        seed=args.seed,
    )
    train_iterator_builder, training_steps, training_samples = \
        cosmoflow_dataset.training_dataset(
            args.training_batch_size,
            args.shard_type,
            args.shuffle,
            args.preshuffle,
            args.training_samples,
            args.data_shard_multiplier,
            args.prestage,
            read_chunk_size=args.read_chunk_size,
        )
    val_iterator_builder, val_steps, val_samples = \
        cosmoflow_dataset.validation_dataset(
            args.validation_batch_size,
            True,
            args.validation_samples,
            read_chunk_size=args.read_chunk_size,
        )

    # MLPerf logging of batch size, and number of samples used in training
    utils.logger.event(key=utils.logger.constants.GLOBAL_BATCH_SIZE,
                       value=args.training_batch_size*dist_desc.size)
    utils.logger.event(
        key=utils.logger.constants.TRAIN_SAMPLES, value=training_samples)
    utils.logger.event(
        key=utils.logger.constants.EVAL_SAMPLES, value=val_samples)

    def iterator_builder():
        utils.logger.start(key='staging_start')
        train_iterator = train_iterator_builder()
        val_iterator = val_iterator_builder()
        utils.logger.end(key='staging_stop')

        return train_iterator, val_iterator

    return (iterator_builder,
            training_steps,
            val_steps)
