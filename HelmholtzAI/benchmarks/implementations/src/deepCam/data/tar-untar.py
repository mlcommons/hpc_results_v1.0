import os
import tarfile
import glob
from mpi4py import MPI


def tarball_files_by_year(dir):
    # create a tarball of the files in each year
    files = os.listdir(dir)

    pass
