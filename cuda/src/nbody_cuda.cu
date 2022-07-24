
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda.h"

#define N 30208
#define BLOCK_SIZE 256
#define DELTA_TIME 0.2f
#define EPS2  1e-9f
#define NUM_TILES (N + BLOCK_SIZE - 1) / BLOCK_SIZE

__shared__ float4 shPosition[BLOCK_SIZE];

// operator overloading for floatX
__device__ float4 operator*(float4 a, float4 b)
{
            float4 result;
            result.x = a.x * b.x;
            result.y = a.y * b.y;
            result.z = a.z * b.z;
            result.w = a.w * b.w;
            return result;
        }

__device__ float4 operator+(float4 a, float4 b)
{
            float4 result;
            result.x = a.x + b.x;
            result.y = a.y + b.y;
            result.z = a.z + b.z;
            result.w = a.w + b.w;
            return result;
}
__device__ float3 operator+(float3 a, float4 b)
{
    float3 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

__device__ float4 operator-(float4 a, float4 b)
{
            float4 result;
            result.x = a.x - b.x;
            result.y = a.y - b.y;
            result.z = a.z - b.z;
            result.w = a.w - b.w;
            return result;
}

__device__ float4 operator*(float4 a, float b)
{
            float4 result;
            result.x = a.x * b;
            result.y = a.y * b;
            result.z = a.z * b;
            result.w = a.w * b;
            return result;
}

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai){
    float4 r;
    // r_ij [3 FLOPS]
    r = bj - bi;
    
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;

    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
    
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai =  ai + r * s;
    
    return ai;
    }

__device__ float3 tile_calculation(float4 myPosition, float3 accel)
{
    int i;

    for (i = 0; i < blockDim.x; i++) {
        accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    }
    return accel;
}

__global__ void calculate_forces(float4* old_pos, float4* old_vel, float4* new_pos, float4* new_vel)
{
    
    float4 myPosition;
    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};
    // global id 
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    
    myPosition = old_pos[gtid];

   

    for (i = 0, tile = 0; i < NUM_TILES; i++, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = old_pos[idx];
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
   

    float4 oldVel;
    oldVel = old_vel[gtid];
    // updated position and velocity
    float4 newPos = myPosition + oldVel * DELTA_TIME + acc4 * 0.5f * DELTA_TIME * DELTA_TIME;
    newPos.w = myPosition.w;
    float4 newVel = oldVel + (acc4 * DELTA_TIME);
    // write to global memory
    new_pos[gtid] = newPos;
    new_vel[gtid] = newVel;
    
}






int main(const int argc, const char** argv) {
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    

    int bytes = N * sizeof(float4);
    float4 *pos = (float4*)malloc(bytes);
    float4 *vel = (float4*)malloc(bytes);

    float4 *new_pos = (float4*)malloc(bytes);
    float4 *new_vel = (float4*)malloc(bytes);

    // Initialization bodies pos and vel
    srand(10);
    for(int i = 0; i < N; i++){
        // pos
        pos[i].x= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i].y= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i].z= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        // mass
        pos[i].w= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        //vel
        vel[i].x= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i].y= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i].z= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i].w= 0;
       
    }

    // device allocation
    float4 *d_old_pos;
    float4 *d_old_vel;

    float4 *d_new_pos;
    float4 *d_new_vel;

    
    
    
    int nBlocks = (N % BLOCK_SIZE!=0) + N / BLOCK_SIZE;

    // Start simulation
    cudaMalloc(&d_old_pos, bytes);
    cudaMalloc(&d_old_vel, bytes);

    cudaMalloc(&d_new_pos, bytes);
    cudaMalloc(&d_new_vel, bytes);

    // Hosto to device
    cudaMemcpy(d_old_pos, pos, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_vel, vel, bytes, cudaMemcpyHostToDevice);

    // Take start time
    cudaEventRecord(start);
    // nbody calculation
    calculate_forces<<<nBlocks, BLOCK_SIZE>>>(d_old_pos, d_old_vel, d_new_pos, d_new_vel); // compute interbody forces
    
    cudaEventRecord(stop);
    // Wait stop event
    cudaEventSynchronize(stop);

    // Take time in ms
    cudaEventElapsedTime(&time, start, stop);
        
    // Device to host
    cudaMemcpy(new_pos, d_new_pos, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_vel, d_new_vel, bytes, cudaMemcpyDeviceToHost);
        
    // print results
    for(int i = 0; i < N; i++){
        printf("body: %d, new_pos_x: %.2f, new_pos_y: %.2f, new_pos_z: %.2f\n", i, new_pos[i].x, new_pos[i].y, new_pos[i].z);
        printf("body: %d, new_vel_x: %.2f, new_vel_y: %.2f, new_vel_z: %.2f\n", i, new_vel[i].x, new_vel[i].y, new_vel[i].z);
        printf("\n");
    }


    cudaFree(d_new_pos);
    cudaFree(d_new_vel);
    cudaFree(d_old_pos);
    cudaFree(d_old_vel);


    printf("Time: %f\n", time);

}