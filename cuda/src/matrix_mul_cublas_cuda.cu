#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

#ifndef MATRIX_SIZE
    #define MATRIX_SIZE 1024
#endif

int main () {
    // For cublas library
    cublasStatus_t stat;
    cublasHandle_t handle;

    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Allocate matrix (see if can be use C++ classes)
    // Host data
    float *A_h = static_cast<float *>(malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE));
    float *B_h = static_cast<float *>(malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE));
    float *C_h = static_cast<float *>(malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE));
	
    // Device data
    float *A_d, *B_d, *C_d; // device data
    cudaMalloc((void **) &A_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMalloc((void **) &B_d, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMalloc((void **) &C_d, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);

    // Data initialization

	// Initialization
    for(int i {0}; i < MATRIX_SIZE * MATRIX_SIZE; i++)
        A_h[i] = 1.0f;
    
    for(int i {0}; i < MATRIX_SIZE * MATRIX_SIZE; i++)
        B_h[i] = 1.0f;
    
    for(int i {0}; i < MATRIX_SIZE * MATRIX_SIZE; i++)
        C_h[i] = 0.0f;
    
    // cuBLAS initialization
    stat = cublasCreate(&handle);              
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    #ifndef KERNEL_TIME
        cudaEventRecord(start);
    #endif
    // auto start = steady_clock::now();
    // Data movement 
    cudaMemcpy(A_d, A_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(C_d, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

   
    // Matrix mul
    const float alpha = 1, beta = 0;
    auto kernel_start = steady_clock::now();
    #ifdef KERNEL_TIME
        // Take start time
        cudaEventRecord(start);
    #endif
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, &alpha, B_d, MATRIX_SIZE, A_d, MATRIX_SIZE, &beta, C_d, MATRIX_SIZE);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Matrix product failed\n");
        cudaFree (A_d);
        cudaFree (B_d);
        cudaFree (C_d);
        cublasDestroy(handle);

        return EXIT_FAILURE;
    }
    cudaDeviceSynchronize();    
    #ifdef KERNEL_TIME
        cudaEventRecord(stop);
        // Wait stop event
        cudaEventSynchronize(stop);
    #endif
    
    auto kernel_end = steady_clock::now();
    
    // Data from device to host
    cudaMemcpy(C_h, C_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    #ifndef KERNEL_TIME
        cudaEventRecord(stop);
        // Wait stop event
        cudaEventSynchronize(stop);
    #endif

    // auto end = steady_clock::now();


    // cout << "matrix_mul_cublas, " << duration_cast<chrono::milliseconds>(end - start).count() << ", " << duration_cast<chrono::microseconds>(kernel_end - kernel_start).count() << "";
    // cout << "matrix_mul_cublas, " << duration_cast<chrono::milliseconds>(kernel_end - kernel_start).count() << std::endl;
    // Take time in ms
    cudaEventElapsedTime(&time, start, stop);
    
    printf("%s, %f\n", "matrix_mul_cublas_cuda", time);
    
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
