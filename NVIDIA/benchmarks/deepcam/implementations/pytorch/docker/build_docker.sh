#!/bin/bash

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

# registry id
private_registry_id="gitlab-master.nvidia.com:5005"
shared_registry_id="nvcr.io/nvdlfwea/mlperfhpc_v10"
#registry_id="registry.services.nersc.gov"

#we need to step out to expand the build context
cd ..

dlfw_versions="21.09"
image_prefix="sanitization"

#dlfw_versions="master"
#image_prefix="debug2"


#training container
for dlfw_version in ${dlfw_versions}; do
    docker build --network host \
	   --build-arg dlfw_version=${dlfw_version} \
	   -t ${private_registry_id}/tkurth/mlperf-deepcam:${image_prefix}-${dlfw_version} -f Dockerfile .
    docker push ${private_registry_id}/tkurth/mlperf-deepcam:${image_prefix}-${dlfw_version}

    ## retag and repush
    #docker tag ${private_registry_id}/tkurth/mlperf-deepcam:${image_prefix}-${dlfw_version} ${shared_registry_id}/deepcam:${image_prefix}-${dlfw_version}
    #docker push ${shared_registry_id}/deepcam:${image_prefix}-${dlfw_version}
    
    #docker build --build-arg dlfw_version=${dlfw_version} -t gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:computelab-${dlfw_version} -f docker/Dockerfile.computelab .
    #docker push gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:computelab-${dlfw_version}

    ## utility container
    #docker build --build-arg dlfw_version=${dlfw_version} -t gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:mxnet-utils-${dlfw_version} -f docker/Dockerfile.mxnet_utils .
    #docker push gitlab-master.nvidia.com:5005/tkurth/mlperf-deepcam:mxnet-utils-${dlfw_version}
done

