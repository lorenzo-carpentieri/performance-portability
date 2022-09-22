#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>


using namespace std;
using namespace chrono;

#ifndef MATRIX_SIZE
    #define MATRIX_SIZE 1024
#endif

#define BLOCK_SIZE 32

// Matrix square with size multiple of BLOCK_SIZE
__global__ void gpu_square_matrix_mult(float *d_a, float *d_b, float *d_result, int n) 
{
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float tmp = 0;
    int idx;
    
    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        // load data in local memory
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        
        tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        
        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        
        tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    
        d_result[row * n + col] = tmp;
}

int main () {
     cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    // Allocate matrix (see if can be use C++ classes)
    // Host data
    float *A_h = static_cast<float *>(malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE));
    float *B_h = static_cast<float *>(malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE));
    float *C_h = static_cast<float *>(malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE));
     // Data initialization

	// Initialization
    for(int i {0}; i < MATRIX_SIZE * MATRIX_SIZE; i++)
        A_h[i] = 1;
    
    for(int i {0}; i < MATRIX_SIZE * MATRIX_SIZE; i++)
        B_h[i] = 1;
    
    for(int i {0}; i < MATRIX_SIZE * MATRIX_SIZE; i++)
        C_h[i] = 0.0f;

    #ifndef KERNEL_TIME
        cudaEventRecord(start);
    #endif

    // Device data
    float *A_d, *B_d, *C_d; // device data
    cudaMalloc((void **) &A_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **) &B_d, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMalloc((void **) &C_d, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);

   
    
    dim3 grid(MATRIX_SIZE/BLOCK_SIZE, MATRIX_SIZE/BLOCK_SIZE); 
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
  
    // Data movement 
    cudaMemcpy(A_d, A_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(C_d, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    #ifdef KERNEL_TIME
    // Take start time
    cudaEventRecord(start);
    #endif
    gpu_square_matrix_mult<<<grid, threads>>>(A_d, B_d,C_d,MATRIX_SIZE);
    
    #ifdef KERNEL_TIME
        cudaEventRecord(stop);
        // Wait stop event
        cudaEventSynchronize(stop);
    #endif

    // Take time in ms
   
    // Data from device to host
    cudaMemcpy(C_h, C_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
   
    cudaDeviceSynchronize();
    #ifndef KERNEL_TIME
        cudaEventRecord(stop);
        // Wait stop event
        cudaEventSynchronize(stop);
    #endif

    cudaEventElapsedTime(&time, start, stop);
    
    printf("%s, %f\n", "matrix_mul_tiling_cuda", time);
    #ifdef DEBUG
    for(int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++){
        if(C_h[i]!=MATRIX_SIZE){
            std::cout<< "fail" << std::endl;
            return -1;
        }
    }
    #endif
    
    // Free device memory
    cudaFree(A_d);     
    cudaFree(B_d);
    cudaFree(C_d);


    
    // Free host memory
    free(A_h);      
    free(B_h);
    free(C_h);

    return EXIT_SUCCESS;
}
