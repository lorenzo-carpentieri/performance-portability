#!/bin/bash
mkdir /root/tesi/performance-portability/sycl/build
cd  /root/tesi/performance-portability/sycl/build

cmake -E env HIPSYCL_TARGETS=cuda:sm_70 cmake ..  -DCMAKE_CXX_COMPILER=syclcc -DKERNEL_PROFILING=1 -DSYCL_BACKEND="hipSYCL" -DSYCL_DEVICE=gpu -DCMAKE_CXX_FLAGS='-O3 -fno-omit-frame-pointer -fno-optimize-sibling-calls'


# cmake ..  -DCMAKE_CXX_COMPILER='clang++' -DSYCL_BACKEND=dpcpp -DSYCL_DEVICE=gpu 	-DCMAKE_CXX_FLAGS='-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_70 -O3  -mcpu=native'


mkdir /root/tesi/performance-portability/cuda/build
cd /root/tesi/performance-portability/cuda/build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_FLAGS=-lcublas -DKERNEL_PROFILING=1
