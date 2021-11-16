# How To Run

This repo holds the codes and configs for running on the JUWELS Booster as well as on HoreKa at KIT. 
To run a trainings simply use the `start_training_run.sh` script from within the `run_scripts` diectory.
This will trigger to other scripts in this order: `start_**_training.sh` and `run_and_time.sh`, where `**` 
is the training system (`jb` for JUWELS Booster and `horeka` for HoreKa).

Example usage:
```bash
./start_training_run --system booster --nodes 34 --time 01:00:00 --config "config_file_path"
```
On JUWELS Booster, a job can be started by calling e.g. `run_cosmoflow_256x4x1.sbatch`. This will queue a 
job with 256 and 4 GPUs per node, and a local batch size of 1.

# Repo Structure

This implementation is based on the implementation of NVIDIA, based on their containers. In our implementation,
we use singularity with an image that is almost identical to NVIDIA's image, except for installations of the packages
h5py and pmi, which we have added to the containers, and then converted to `.sif` files for singularity. We have copied
NVIDIA's python code for the benchmark out of the container and adjusted it to our needs. This modified code is found
in the directory `cosmoflow`. 

In the directory, `hdf5_io` the our conversion method to h5 is implemented. 

# HDF5 IO
We use hdf5 to store the input data. The conversion is implemented in `convert_cosmoflow_to_hdf5.py`. In summary, this method creates
large `.h5` files for training and validation. 



