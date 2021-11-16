import os
import sys
import glob
import h5py as h5
import numpy as np
import math
import argparse as ap
import mxnet as mx
from mpi4py import MPI

def filter_func(item, lst):
    item = os.path.basename(item).replace(".h5", ".npy")
    return item not in lst


def read(ifname):
    with h5.File(ifname, 'r') as f:
        data = f["climate/data"][...]
        label = f["climate/labels_0"][...]
        
    return data, label
        

def main(args):

    # get rank
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    # get input files
    inputfiles_all = glob.glob(os.path.join(args.input_directory, "*.h5"))

    # select just a few files
    if pargs.num_files is not None:
        num_files = max([min([len(inputfiles_all), pargs.num_files]), 0])
        inputfiles_all = inputfiles_all[:num_files]

    # create output dir
    output_dir = pargs.output_directory
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # create recordio files
    data_record = mx.recordio.MXIndexedRecordIO(os.path.join(output_dir, 'data.idx'), os.path.join(output_dir, 'data.rec'), 'w')
    label_record = mx.recordio.MXIndexedRecordIO(os.path.join(output_dir, 'label.idx'), os.path.join(output_dir, 'label.rec'), 'w')

    for idx, filename in enumerate(inputfiles_all):
        # read file
        data, label = read(filename)
        
        # create header
        header = mx.recordio.IRHeader(0, 0., idx, 0)

        # pack
        data_packed = mx.recordio.pack(header, data.tobytes())
        label_packed = mx.recordio.pack(header, label.tobytes())
        
        # write:
        data_record.write_idx(idx, data_packed)
        label_record.write_idx(idx, label_packed)
    
    # wait for the others
    comm.barrier()


if __name__ == "__main__":
    
    AP = ap.ArgumentParser()
    AP.add_argument("--input_directory", type=str, help="Directory with input files", required = True)
    AP.add_argument("--output_directory", type=str, help="Directory for output files", required = True)
    AP.add_argument("--num_files", type=int, default=None, help="Maximum number of files to convert")
    pargs = AP.parse_args()
    
    main(pargs)
