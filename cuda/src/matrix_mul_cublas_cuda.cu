#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

//#define DEBUG

int main (int argc, char ** argv) {
    // For cublas library
    cublasStatus_t stat;
    cublasHandle_t handle;
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
        A_h[i] = 1.0f;
    
    for(int i {0}; i < N * N; i++)
        B_h[i] = 1.0f;
    
    for(int i {0}; i < N * N; i++)
        C_h[i] = 0.0f;
    
    // cuBLAS initialization
    stat = cublasCreate(&handle);              
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    auto start = steady_clock::now();
    // Data movement 
    cudaMemcpy(A_d, A_h, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(C_d, 0, N * N * sizeof(float));

   
    // Matrix mul
    const float alpha = 1, beta = 0;
    auto kernel_start = steady_clock::now();
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B_d, N, A_d, N, &beta, C_d, N);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Matrix product failed\n");
        cudaFree (A_d);
        cudaFree (B_d);
        cudaFree (C_d);
        cublasDestroy(handle);

        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();
    auto kernel_end = steady_clock::now();
    
    // Data from device to host
    cudaMemcpy(C_h, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    auto end = steady_clock::now();


    cout << duration_cast<chrono::milliseconds>(end - start).count() << ", " << duration_cast<chrono::microseconds>(kernel_end - kernel_start).count() << "";
    
    // Free device memory
    cudaFree(A_d);     
    cudaFree(B_d);
    cudaFree(C_d);

    // Destroy cublas context
    cublasDestroy(handle);  
    
    // Free host memory
    free(A_h);      
    free(B_h);
    free(C_h);

    return EXIT_SUCCESS;
}
