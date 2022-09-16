#!/bin/bash
cd /root/loren/tesi/sycl/build
cmake .. -DCMAKE_CXX_COMPILER=syclcc -DSYCL_BACKEND=hipSYCL -DSYCL_DEVICE=gpu


cd /root/loren/tesi/cuda/build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lcublas
