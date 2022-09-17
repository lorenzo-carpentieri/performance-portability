/*
    Baseline
*/
#include "cuda_runtime.h"
#include "cuda.h"
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>


#define T float

#define BLOCK_SIZE 512
#define N 30720
#define N_BLOCKS (N/BLOCK_SIZE)

__device__ __forceinline__ void unrolling(volatile T* local_data, int thread_id) {
    
    if (BLOCK_SIZE >= 64) local_data[thread_id] += local_data[thread_id + 32];
    if (BLOCK_SIZE >= 32) local_data[thread_id] += local_data[thread_id + 16];
    if (BLOCK_SIZE >= 16) local_data[thread_id] += local_data[thread_id + 8];
    if (BLOCK_SIZE >= 8) local_data[thread_id] += local_data[thread_id + 4];
    if (BLOCK_SIZE >= 4) local_data[thread_id] += local_data[thread_id + 2];
    if (BLOCK_SIZE >= 2) local_data[thread_id] += local_data[thread_id + 1];
}

__global__ void reduction(T* input, T* output)
{
    __shared__ T local_data[BLOCK_SIZE];
    
    int grid_size = N_BLOCKS * BLOCK_SIZE;
    T my_sum = 0;

    // In this case we also halve the number of blocks
    // threads in block0 sum data in block_0 and block_1
    // threads in block_1 sum data in block_2 and block_3
    // ...
    // starting index 
    unsigned int i = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    grid_size = grid_size << 1;
    while (i < N) {
        my_sum += input[i];
           
        if ((i + BLOCK_SIZE) < N)
            my_sum += input[i + BLOCK_SIZE];

        i += grid_size;
    }
   
    // each thread puts its local sum into shared memory
    local_data[threadIdx.x] = my_sum;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) // Only the first half of the threads do the computation
            local_data[threadIdx.x] += local_data[threadIdx.x + stride];

        __syncthreads(); // Wait that all threads in the block compute the partial sum
    }


    if(threadIdx.x < 32)
        unrolling(local_data, threadIdx.x);

    
    if (threadIdx.x == 0) 
        atomicAdd(output, local_data[0]);
}


int main()
{
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    T *h_input = (T *) malloc(N * sizeof(T));
    T *h_output = (T *) malloc(sizeof(T));

     

    if (!h_input) // Check if malloc was all right
        return -1;

    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;

    // Allocating memory for device
    T* d_input, *d_output;
    cudaMalloc((void **) &d_input, N * sizeof(T));
    cudaMalloc((void**) &d_output, sizeof(T));

    // Copying input data from host to device, executing kernel and copying result from device to host

    *h_output = 0;
    cudaEventRecord(start);

    cudaMemcpy(d_input, h_input, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, sizeof(T), cudaMemcpyHostToDevice);
        
    reduction<<<N_BLOCKS, BLOCK_SIZE>>> (d_input, d_output);
  

    cudaMemcpy(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);
     cudaEventRecord(stop);
    // Wait stop event
    cudaEventSynchronize(stop);
    // Take time in ms
    cudaEventElapsedTime(&time, start, stop);
    printf("%s, %f\n", "reduction_binary_cuda", time);
    if (*h_output == N)
        printf("pass\n");
    else
        printf("fail, result: %f\n", *h_output);

    cudaDeviceSynchronize();
    


    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}