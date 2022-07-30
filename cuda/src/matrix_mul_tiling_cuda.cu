#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>


using namespace std;
using namespace chrono;

#define BLOCK_SIZE 2

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

int main (int argc, char ** argv) {

    size_t N;

    if(argc != 2) {
        fprintf(stderr, "Usage: %s <N> \n", argv[0]);

        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);


    // Allocate matrix (see if can be use C++ classes)
    // Host data
    float *A_h = static_cast<float *>(malloc(sizeof(float) * N * N));
    float *B_h = static_cast<float *>(malloc(sizeof(float) * N * N));
    float *C_h = static_cast<float *>(malloc(sizeof(float) * N * N));
	
    // Device data
    float *A_d, *B_d, *C_d; // device data
    cudaMalloc((void **) &A_d, N * N * sizeof(float));
    cudaMalloc((void **) &B_d, sizeof(float) * N * N);
    cudaMalloc((void **) &C_d, sizeof(float) * N * N);

    // Data initialization

	// Initialization
    for(int i {0}; i < N * N; i++)
        A_h[i] = i+1;
    
    for(int i {0}; i < N * N; i++)
        B_h[i] = i+1;
    
    for(int i {0}; i < N * N; i++)
        C_h[i] = 0.0f;
    
    dim3 grid(N/BLOCK_SIZE, N/BLOCK_SIZE); 
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
  
    // Data movement 
    cudaMemcpy(A_d, A_h, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(C_d, 0, N * N * sizeof(float));

     gpu_square_matrix_mult<<<grid, threads>>>(A_d, B_d,C_d,N);

    
    // Data from device to host
    cudaMemcpy(C_h, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < N*N; i++){
        std::cout << C_h[i] << ", ";
    }
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
