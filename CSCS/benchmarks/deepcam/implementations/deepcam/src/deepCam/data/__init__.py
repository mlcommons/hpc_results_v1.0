import os
from glob import glob

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from .cam_hdf5_dataset import CamDataset, CachedDataset, peek_shapes_hdf5

# helper function for determining the data shapes
def get_datashapes(pargs, root_dir):
    
    return peek_shapes_hdf5(os.path.join(root_dir, "train"))
    

# helper function to de-clutter the main training script
def get_train_dataloader(pargs, root_dir, device, seed, comm_size, comm_rank):
    
    # import only what we need
    train_dir = os.path.join(root_dir, "train")
    train_set = CamDataset(train_dir, n_samples = pargs.train_samples,
                           statsfile = os.path.join(root_dir, 'stats.h5'),
                           channels = pargs.channels,
                           allow_uneven_distribution = True,
                           shuffle = True, 
                           preprocess = True,
                           comm_size = comm_size if pargs.train_dataset_sharding else 1,
                           comm_rank = comm_rank if pargs.train_dataset_sharding else 0,
                           seed = seed)

    # Wrap CamDataset with CachedDataset and cache to RAM
    if pargs.train_dataset_caching: # suitable at comm_size > 200
        train_set = CachedDataset(train_set)

    if pargs.train_dataset_sharding:
        sampler_shuffle_opts = dict(shuffle=True)
    else:
        sampler = DistributedSampler(train_set,
                                     num_replicas = comm_size,
                                     rank = comm_rank,
                                     shuffle = True,
                                     drop_last = True)
        sampler_shuffle_opts = dict(sampler=sampler)
    
    train_loader = DataLoader(train_set,
                              pargs.local_batch_size, 
                              num_workers = pargs.max_inter_threads,
                              **sampler_shuffle_opts,
                              pin_memory = True,
                              drop_last = True,
                              persistent_workers = True)

    train_size = train_set.global_size
    return train_loader, train_size


def get_val_dataloader(pargs, root_dir, device, seed, comm_size, comm_rank):
    validation_dir = os.path.join(root_dir, "validation")
    validation_set = CamDataset(validation_dir, n_samples = pargs.valid_samples,
                                statsfile = os.path.join(root_dir, 'stats.h5'),
                                channels = pargs.channels,
                                allow_uneven_distribution = True,
                                shuffle = False,
                                preprocess = True,
                                comm_size = comm_size,
                                comm_rank = comm_rank)

    # Wrap CamDataset with CachedDataset (starting at smaller node count than training)
    if pargs.valid_dataset_caching: # suitable at comm_size > 20
        validation_set = CachedDataset(validation_set)

    validation_loader = DataLoader(validation_set,
                                   pargs.valid_batch_size,
                                   num_workers = pargs.max_inter_threads,
                                   pin_memory = True,
                                   drop_last = False,
                                   persistent_workers = True)
    
    validation_size = validation_set.global_size
        
    return validation_loader, validation_size
