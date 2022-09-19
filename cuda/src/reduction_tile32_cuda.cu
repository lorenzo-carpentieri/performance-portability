/*
    Using cooperative groups to partition each block in tiles of 32 threads
    While halving the tiles in each block
*/

#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define T float
#define BLOCK_SIZE 512

#ifndef SIZE_REDUCTION    
    #define SIZE_REDUCTION 30720 
#endif
#define N_BLOCKS (SIZE_REDUCTION/BLOCK_SIZE)
#define NUM_TILES_PER_BLOCK (BLOCK_SIZE / 32)

using namespace cooperative_groups;

__device__ void sum32(thread_block_tile<32> tile, volatile T* array) {

    int thread_id = tile.thread_rank(); // Returns the id of the calling threads in the tile(between 0 and 31)
    
    int i = tile.meta_group_rank() * 32 * 2 + thread_id;

    // We work on a warp (32-threads)
    if (i < BLOCK_SIZE) {
        array[i] += array[i + 32];
        array[i] = reduce(tile, array[i], plus<T>());
    }
}

__global__ void reduction(T* input, T* output)
{
    __shared__ T local_data[BLOCK_SIZE];
    thread_block block = this_thread_block();

    int thread_id = block.thread_rank();
    int block_id = block.group_index().x;
    int grid_size = N_BLOCKS * BLOCK_SIZE;

    T my_sum = 0;

    // In this case we also halve the number of blocks

    unsigned int i = block_id * BLOCK_SIZE * 2 + thread_id;
    grid_size = grid_size << 1;
    while (i < SIZE_REDUCTION) {
        my_sum += input[i];
        if ((i + BLOCK_SIZE) < SIZE_REDUCTION) {
            my_sum += input[i + BLOCK_SIZE];
        }
        i += grid_size;
    }

    // each thread puts its local sum into shared memory
    local_data[thread_id] = my_sum;
    block.sync(); // Wait that all threads load data from global memory to local memory

    // We are partitioning each block in tiles of 32 threads
    thread_block_tile<32> tile32 = tiled_partition<32>(block);
    sum32(tile32, local_data);

    int j = tile32.meta_group_rank() * 32 * 2 + tile32.thread_rank();

    if (tile32.thread_rank() == 0 && tile32.meta_group_rank() < NUM_TILES_PER_BLOCK / 2)
        atomicAdd(output, local_data[j]);
}

int main()
{
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    T* h_input = (T*)malloc(SIZE_REDUCTION * sizeof(T));
    T* h_output = (T*)malloc(sizeof(T));


    if (!h_input) // Check if malloc was all right
        return -1;

    for (int i = 0; i < SIZE_REDUCTION; i++)
        h_input[i] = 1.0f;

    // Allocating memory for device
    T* d_input, * d_output;
    cudaMalloc((void**)&d_input, SIZE_REDUCTION * sizeof(T));
    cudaMalloc((void**)&d_output, sizeof(T));

    // Copying input data from host to device, executing kernel and copying result from device to host

    *h_output = 0;
    
    cudaMemcpy(d_input, h_input, SIZE_REDUCTION * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, sizeof(T), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    reduction << <N_BLOCKS, BLOCK_SIZE >> > (d_input, d_output);
    
    cudaEventRecord(stop);
    // Wait stop event
    cudaEventSynchronize(stop);
    // Take time in ms
    cudaEventElapsedTime(&time, start, stop);
    printf("%s, %f\n", "reduction_tile32_cuda", time);

    cudaMemcpy(h_output, d_output, sizeof(T), cudaMemcpyDeviceToHost);


    #ifdef DEBUG
    if (*h_output == SIZE_REDUCTION)
        printf("pass\n");
    else
        printf("fail, result: %f\n", *h_output);
    #endif

    cudaDeviceSynchronize();
    


    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}