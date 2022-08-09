#include <stdio.h>
#include <iostream>
// kernels transpose/copy a tile of TILE_DIM x TILE_DIM elements
// using a TILE_DIM x BLOCK_ROWS thread block, so that each thread
// transposes TILE_DIM/BLOCK_ROWS elements. TILE_DIM must be an
// integral multiple of BLOCK_ROWS
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define SIZE_X 4096
#define SIZE_Y 4096

// Number of repetitions used for timing.
#define NUM_REPS 100

// width, height matrix dimensions
__global__ void transposeCoalesced(float *odata, float *idata, int width, int height)
{
    // initialize local memory
    // TILE_DIM+1 to avoid bank conflicts 
    __shared__ float tile[TILE_DIM][TILE_DIM+1];
    int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
    
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    
    // Copy data in local memory
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }
    
    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
}



int main( int argc, char** argv)
{
    
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Computation is divided into tiles of TILE_DIM X TILE_DIME (where TILE_DIM is multiple of BLOCK_ROWS). 
    // execution configuration parameters
    dim3 grid(SIZE_X/TILE_DIM, SIZE_Y/TILE_DIM); 
    dim3 threads(TILE_DIM,BLOCK_ROWS);
    
 
    
    // size of memory required to store the matrix
    const int mem_size = sizeof(float) * SIZE_X*SIZE_Y;
    
    // allocate host memory
    float *h_idata = (float*) malloc(mem_size);
    float *h_odata = (float*) malloc(mem_size);
    // allocate device memory
    float *d_idata, *d_odata;
    cudaMalloc( (void**) &d_idata, mem_size);
    cudaMalloc( (void**) &d_odata, mem_size);
    // initalize host data
    for(int i = 0; i < (SIZE_X*SIZE_Y); ++i)
        h_idata[i] = (float) i;
    
    // copy host data to device
    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice );

      // set reference solution
    // NB: fine- and coarse-grained kernels are not full
    // transposes, so bypass check
    
    // initialize events, EC parameters
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // take measurements for loop over kernel launches
    cudaEventRecord(start, 0);
    transposeCoalesced<<<grid, threads>>>(d_odata, d_idata,SIZE_X,SIZE_Y);
   
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(h_odata,d_odata, mem_size, cudaMemcpyDeviceToHost);
    
    printf("%s, %f\n", "matrix_transpose_cuda", time);

    #ifdef DEBUG
    for(int i = 0; i < SIZE_X*SIZE_Y; i++)
        std::cout << h_odata[i] << ", "; 
    #endif

    // cleanup
    free(h_idata); 
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}