# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import torch
import torch.distributed as dist

def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    return rank


def get_local_rank():
    """
    Gets node local rank or returns zero if distributed is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    
    #number of GPUs per node
    if torch.cuda.is_available():
        local_rank = dist.get_rank() % torch.cuda.device_count()
    else:
        local_rank = 0
        
    return local_rank


def get_size():
    """
    Gets size of communicator
    """
    if dist.is_available() and dist.is_initialized():
        size = dist.get_world_size()
    else:
        size = 1
    return size


def init(method):
    #get master address and port
    print("comm init method", method)
    if method == "nccl-openmpi":
        addrport = os.getenv("PMIX_SERVER_URI2").split("//")[1]
        #use that URI
        address = addrport.split(":")[0]
        #use the default pytorch port
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port
        rank = int(os.getenv('OMPI_COMM_WORLD_RANK',0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE",0))
        print(addrport, rank, world_size)
        #init DDP
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)
        
    elif method == "nccl-slurm":
        rank = int(os.getenv("PMIX_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        #init DDP
        print(rank, world_size, address)
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)

    elif method == "nccl-slurm-pmi":
        rank = int(os.getenv("PMI_RANK"))
        world_size = int(os.getenv("SLURM_NTASKS"))
        address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
        port = "29500"
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port
                                                
        #init DDP
        dist.init_process_group(backend = "nccl",
                                rank = rank,
                                world_size = world_size)
        
    elif method == "mpi":
        #init DDP
        dist.init_process_group(backend = "mpi")
        
    else:
        raise NotImplementedError()
