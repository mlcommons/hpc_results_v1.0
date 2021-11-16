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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.09-py3
FROM ${FROM_IMAGE_NAME}

ARG dlfw_version=21.09
ARG dlfw_version
ENV DLFW_VERSION ${dlfw_version}				    

#install mpi4py
RUN pip install h5py mpi4py

# pyprof
RUN pip install nvidia-pyindex && \
    pip install nvidia-dlprof[pytorch]

#pip install more python modules
RUN pip install wandb

#install mlperf logging
RUN pip install "git+https://github.com/mlcommons/logging.git@hpc-1.0.0-rc1"

#install gradual warmup
RUN pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

#install io_helpers
COPY io_helpers /opt/io_helpers
RUN cd /opt/io_helpers && python setup.py install

#copy main scripts
COPY src/deepCam /opt/deepCam
COPY src/utils /opt/utils
COPY cleanup.sh /opt/deepCam/cleanup.sh

# worker scripts and files
COPY run_and_time.sh /workspace/run_and_time.sh
COPY init_datasets.sub /workspace/init_datasets.sub
COPY run.sub /workspace/run.sub
COPY configs /workspace/configs

#init empty git repo so that wandb works
RUN cd /opt/deepCam && git init

#create additional folders for mapping data in
RUN mkdir -p /data

