#include <iostream>
#include <cstdlib>
#include "../include/lodepng.h"
#include <cuda_runtime.h>
#include <functional>
#include <cmath>

// #define R           3
// #define D           (R*2+1)
// #define S           (D*D)
// #define BLOCK_W     32
// #define BLOCK_H     32
// #define IMG_SIZE 512

// __global__ void blur(const unsigned char *in_r, const unsigned char *in_g, const unsigned char *in_b, unsigned char *out, const unsigned int w, const unsigned int h){
//     //shared mem
//     __shared__ float smem_r[BLOCK_W][BLOCK_H];
//     __shared__ float smem_g[BLOCK_W][BLOCK_H];
//     __shared__ float smem_b[BLOCK_W][BLOCK_H];
    

//     int threadsPerBlock  = blockDim.x * blockDim.y;
//     int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
//     int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
//     const int x = threadIdx.x;
//     const int y = threadIdx.y;
//     const int gidx	= blockIdx.x * blockDim.x + x;
// 	const int gidy	= blockIdx.y * blockDim.y + y;

//     //Indexes
//     int gid = blockNumInGrid * threadsPerBlock + threadNumInBlock;
//     // offset to load correct value in local memory 
//     int offset = (blockIdx.y * BLOCK_W + y)*w + x + blockIdx.x * BLOCK_H;
//     // each thread in a block load a data in the local memory
//     smem_r[threadIdx.y][threadIdx.x] = in_r[(gidy*w+gidx)];
//     smem_g[threadIdx.y][threadIdx.x] = in_g[(gidy*w+gidx)];
//     smem_b[threadIdx.y][threadIdx.x] = in_b[(gidy*w+gidx)];
//     __syncthreads();
    
//     // box filter (only for threads inside the tile)
         
//         float sum_r = 0;
//         float sum_g = 0;
//         float sum_b = 0;
//         int hits = 0;
//         for(int dx = -R; dx < R; dx++){
//          for(int dy = -R; dy <R; dy++)
//              if((y + dx) > -1 && (y+dx) < BLOCK_W && (x+dy) > -1 && (x+dy) < BLOCK_H) {
//                  sum_r += smem_r[y + dx][x + dy];
//                  sum_g += smem_g[y + dx][x + dy];
//                  sum_b += smem_b[y + dx][x + dy];
//                  hits++;
//              }
//              else{
//                  sum_r += smem_r[y][x];
//                  sum_g += smem_g[y][x];
//                  sum_b += smem_b[y][x];
//                  hits++;
//              }
//          }
        
    
//         out[offset * 3  ] = sum_r / hits;
//         out[offset * 3+1] = sum_g / hits;       
//         out[offset * 3+2] = sum_b / hits; 
// }

#define R              5
#define D              (R*2+1)
#define S              (D*D)
#define BLOCK_SIZE     512
#define IMG_SIZE       512

// kernel with just one dimension
__global__ void monodimensional_blur(const unsigned char *in_r, const unsigned char *in_g, const unsigned char *in_b, unsigned char *out, const unsigned int w, const unsigned int h){
    const int NUM_BLOCKS = w*h / IMG_SIZE;
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = gidx % w;
    const int y = (gidx-x)/w;
 
    //shared mem
    __shared__ unsigned char smem_r[D][BLOCK_SIZE];
    __shared__ unsigned char smem_g[D][BLOCK_SIZE];
    __shared__ unsigned char smem_b[D][BLOCK_SIZE];

    int row = 0;
    
    //load data in shared memory: each thread load in shared memory the upper and lower row 
    for(int i = -R; i < R + 1; i++){
            if((static_cast<int>(blockIdx.x) + i) > -1 && (static_cast<int>(blockIdx.x) + i) < NUM_BLOCKS){
                smem_r[row][threadIdx.x] = in_r[gidx+(i*w)];
                smem_g[row][threadIdx.x] = in_g[gidx+(i*w)];
                smem_b[row][threadIdx.x] = in_b[gidx+(i*w)];
            }
            else{
                smem_r[row][threadIdx.x] = 0;
                smem_g[row][threadIdx.x] = 0;
                smem_b[row][threadIdx.x] = 0;
            }
            row++;
    }
    __syncthreads();
    
        // box filter (only for threads inside the tile)
         
        float sum_r = 0;
        float sum_g = 0;
        float sum_b = 0;
        int hits = 0;
       
        for(int ox = -R; ox < R+1; ++ox) {
            for(int oy = -R; oy < R+1; ++oy) {
                if((x+ox) > -1 && (x+ox) < w && (y+oy) > -1 && (y+oy) < h) {
                    sum_r += smem_r[R+ox][x+oy]; 
                    sum_g += smem_g[R+ox][x+oy];
                    sum_b += smem_b[R+ox][x+oy];
                    hits++;
                }
            }
        }
        out[gidx*3  ] = sum_r / hits;
        out[gidx*3+1] = sum_g / hits;       
        out[gidx*3+2] = sum_b / hits; 
}


void filter (unsigned char* input_r,unsigned char* input_g,unsigned char* input_b, unsigned char* output_image, int width, int height) {
     cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned char* dev_r;
    unsigned char* dev_g;
    unsigned char* dev_b;

    unsigned char* dev_output;
    // Memory for red
    cudaMalloc( (void**) &dev_r, width*height*sizeof(unsigned char));
    cudaMemcpy( dev_r, input_r, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice );
    // Memory for green
    cudaMalloc( (void**) &dev_g, width*height*sizeof(unsigned char));
    cudaMemcpy( dev_g, input_g, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice );
    // Memory for blue
    cudaMalloc( (void**) &dev_b, width*height*sizeof(unsigned char));
    cudaMemcpy( dev_b, input_b, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice );
    // Output
    cudaMalloc( (void**) &dev_output, width*height*3*sizeof(unsigned char));

   
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid(height*width/BLOCK_SIZE) ;
    // Take start time
    cudaEventRecord(start);
    monodimensional_blur<<<blocksPerGrid, threadsPerBlock>>>(dev_r, dev_g, dev_b, dev_output, width, height); 
    
    cudaEventRecord(stop);
    // Wait stop event
    cudaEventSynchronize(stop);

    // Take time in ms
    cudaEventElapsedTime(&time, start, stop);
    
    printf("%s, %f\n", "box_blur_local_memory_cuda", time);
   

    cudaMemcpy(output_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost );

    cudaFree(dev_r);
    cudaFree(dev_g);
    cudaFree(dev_b);
    
    cudaFree(dev_output);

}


int main(int argc, char** argv) {
    if(argc != 3) {
        std::cout << "Run with input and output image filenames." << std::endl;
        return 0;
    }

    // Read the arguments
    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::vector<unsigned char> in_image;
    unsigned int width, height;

    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    // Prepare the data
    unsigned char* input_image = new unsigned char[width*height*3];
    unsigned char* output_image = new unsigned char[width*height*3];

    // After image loading I take just the color rgb without the alpha channel  
    int where = 0;
    for(int i = 0; i < in_image.size(); i++) {
        // skip the alpha channel
       if((i+1) % 4 != 0) {
           input_image[where] = in_image.at(i);
           output_image[where] = in_image.at(i);
           where++;
       }
    }
    
    unsigned char* input_r = new unsigned char[width*height];
    unsigned char* input_g = new unsigned char[width*height];
    unsigned char* input_b = new unsigned char[width*height];


    for(int i = 0; i < width * height; i++){
        input_r[i] = input_image[i*3];
        input_g[i] = input_image[i*3+1];
        input_b[i] = input_image[i*3+2];
    }
    


    // Run the filter on it
    // filter(input_image, output_image, width, height); 
    filter(input_r, input_g, input_b, output_image, width, height); 



    // Prepare data for output
    // Add alpha channel in out image
    std::vector<unsigned char> out_image;
    for(int i = 0; i < width*height*3; i++) {
        out_image.push_back(output_image[i]);
        if((i+1) % 3 == 0) {
            out_image.push_back(255);
        }
    }
    
    // Output the data
    error = lodepng::encode(output_file, out_image, width, height);

    //if there's an error, display it
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    delete[] input_image;
    delete[] output_image;
    return 0;

}



