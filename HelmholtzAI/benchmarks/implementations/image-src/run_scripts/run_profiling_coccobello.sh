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

#!/bin/bash

#ranks per node
rankspernode=1
totalranks=${rankspernode}

#env
export OMPI_MCA_btl=^openib

#profile base
#profilebase="$(which nv-nsight-cu-cli) --profile-from-start off -f"
#profilebase="$(which nsys) profile -f true --trace nvtx,cuda,cublas"
#profilebase="$(which nsys) profile -t cublas,cudnn,cuda,nvtx -s none  -f true --export sqlite --capture-range=cudaProfilerApi"
#profilebase='dlprof --nsys_opts="-t cublas,cudnn,cuda,nvtx -s none  -f true --export sqlite --capture-range=cudaProfilerApi"'

#mpi stuff
mpioptions="--allow-run-as-root"

#parameters
run_tag="deepcam_profile"
data_dir_prefix="/data"
output_dir="${data_dir_prefix}/profiles/${run_tag}"

# profile cmd
#profilecmd="${profilebase} -o ${output_dir}/deepcam_profile_${DLFW_VERSION}"
#profilecmd=""

#create files
mkdir -p ${output_dir}

mpirun -np ${totalranks} ${mpioptions} ../bind.sh --cpu=exclusive \
       ${profilecmd} \
       $(which python) ../train.py \
       --wireup_method "dummy" \
       --run_tag ${run_tag} \
       --data_dir_prefix ${data_dir_prefix} \
       --output_dir ${output_dir} \
       --optimizer "LAMB" \
       --start_lr 0.0055 \
       --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \
       --lr_warmup_steps 400 \
       --lr_warmup_factor 1. \
       --weight_decay 1e-2 \
       --logging_frequency 1 \
       --save_frequency 10000 \
       --max_epochs 10 \
       --max_inter_threads 4 \
       --seed 333 \
       --data_format "dali-numpy" \
       --precision_mode "amp" \
       --enable_nhwc \
       --enable_jit \
       --disable_validation \
       --local_batch_size 2
