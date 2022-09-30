#include <iostream>
#include <cstdlib>
#include "../include/lodepng.h"
#include <cuda_runtime.h>
#include <functional>
#include <cmath>



#ifndef RADIUS_SOBEL
    #define RADIUS_SOBEL 3
#endif
__device__ float clamp(float x, float a, float b)
{
  return max(a, min(b, x));
}
__global__
void sobel_filter(float3* in, float3* out, int* size, float* kernel) {
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int gidx	= blockIdx.x * blockDim.x + x;
	const int gidy	= blockIdx.y * blockDim.y + y;


    float3 Gx = float3{0, 0, 0};
    float3 Gy = float3{0, 0, 0};
    // constant-size loops in [0,1,2]
    for(int x_shift = 0; x_shift < 3; x_shift++) {
        for(int y_shift = 0; y_shift < 3; y_shift++) {
             // sample position
             int xs = gidy + x_shift - 1; // [x-1,x,x+1]
             int ys = gidx + y_shift - 1; // [y-1,y,y+1]
             // for the same pixel, convolution is always 0
             if(gidy == xs && gidx == ys)
             continue;
             // boundary check
             if(xs < 0 || xs >= *size || ys < 0 || ys >= *size)
             continue;

            // sample color
            float3 sample = in[xs * (*size)+ys];
            // convolution calculation
            int offset_x = x_shift + y_shift * RADIUS;
            int offset_y = y_shift + x_shift * RADIUS;

            float conv_x = kernel[offset_x];

            Gx.x= Gx.x + conv_x * sample.x;
            Gx.y= Gx.y + conv_x * sample.y;
            Gx.z= Gx.z + conv_x * sample.z;

            float conv_y = kernel[offset_y];

            Gy.x= Gy.x + conv_y * sample.x;
            Gy.y= Gy.y + conv_y * sample.y;
            Gy.z= Gy.z + conv_y * sample.z;
        }
    }

            // taking root of sums of squares of Gx and Gy
            float3 color{0,0,0};
            color.x = hypotf(Gx.x, Gy.x);
            color.y = hypotf(Gx.y, Gy.y);
            color.z = hypotf(Gx.z, Gy.z);

            float3 minval = float3{0.0, 0.0, 0.0};
            float3 maxval = float3{1.0, 1.0, 1.0};

            out[gidx+gidy*(*size)].x = clamp(color.x, minval.x, maxval.x);
            out[gidx+gidy*(*size)].y = clamp(color.y, minval.x, maxval.x);
            out[gidx+gidy*(*size)].z = clamp(color.z, minval.x, maxval.x);            
}

void filter (float3* input_image, float3* output_image, const int size) {
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float3* dev_input;
    float3* dev_output;
    float *dev_kernel;
    int *dev_size;
    const float kernel [] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    
    #ifndef KERNEL_TIME
        cudaEventRecord(start);
    #endif

    cudaMalloc( (void**) &dev_input, size*size*sizeof(float3));
    cudaMalloc( (void**) &dev_kernel, sizeof(float)*9);
    cudaMalloc( (void**) &dev_size, sizeof(int));

    cudaMemcpy( dev_input, input_image, size*size*sizeof(float3), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_kernel, kernel, sizeof(float)*9, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_size, &size, sizeof(int), cudaMemcpyHostToDevice );


    cudaMalloc( (void**) &dev_output, size*size*sizeof(float3));


    dim3 blockDims(32,32);
    dim3 gridDims(size/32,size/32);
    #ifdef KERNEL_TIME
        // Take start time
        cudaEventRecord(start);
    #endif
    sobel_filter<<<gridDims, blockDims>>>(dev_input, dev_output, dev_size, dev_kernel); 

    #ifdef KERNEL_TIME
        cudaEventRecord(stop);
        // Wait stop event
         cudaEventSynchronize(stop);
    #endif
    
    cudaMemcpy(output_image, dev_output, size*size*sizeof(float3), cudaMemcpyDeviceToHost);
    #ifndef KERNEL_TIME
        cudaEventRecord(stop);
        // Wait stop event
         cudaEventSynchronize(stop);
    #endif
    // Take time in ms
    cudaEventElapsedTime(&time, start, stop);
    printf("%s, %f\n", "sobel_filter_cuda", time);

    cudaFree(dev_input);
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

    float3* output_rgb = (float3*) malloc(sizeof(float3)* width*height);


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

    float3* input_rgb = (float3*) malloc(sizeof(float3) * width*height);


    for(int i = 0; i < width * height; i++)
        input_rgb[i] = {input_image[i*3]/255.f, input_image[i*3+1]/255.f, input_image[i*3+2]/255.f};


    // Run the filter on it
    filter(input_rgb, output_rgb, width);

    for(int i = 0; i < width * height; i++){
        int j = i*3;
        output_image[j] = output_rgb[i].x*255.f;
        output_image[j+1] = output_rgb[i].y*255.f;
        output_image[j+2] = output_rgb[i].z*255.f;
    }




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
