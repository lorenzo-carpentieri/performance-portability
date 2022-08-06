#include <iostream>
#include <cstdlib>
#include "../include/lodepng.h"
#include <cuda_runtime.h>
#include <functional>
#include <cmath>

__global__
void blur(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset-x)/width;
    int fsize = 5; // Filter size
    if(offset < width*height) {

        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;
        int hits = 0;
        for(int ox = -fsize; ox < fsize+1; ++ox) {
            for(int oy = -fsize; oy < fsize+1; ++oy) {
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    const int currentoffset = (offset+ox+oy*width)*3;
                    output_red += input_image[currentoffset]; 
                    output_green += input_image[currentoffset+1];
                    output_blue += input_image[currentoffset+2];
                    hits++;
                }
            }
        }
          
        output_image[offset*3] = output_red/hits;
        output_image[offset*3+1] = output_green/hits;
        output_image[offset*3+2] = output_blue/hits;
        }
}

void filter (unsigned char* input_image, unsigned char* output_image, int width, int height) {

    unsigned char* dev_input;
    unsigned char* dev_output;
    cudaMalloc( (void**) &dev_input, width*height*3*sizeof(unsigned char));
    cudaMemcpy( dev_input, input_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice );

    cudaMalloc( (void**) &dev_output, width*height*3*sizeof(unsigned char));

    printf("width: %d, height: %d\n", width, height);
  
    dim3 blockDims(512,1,1);
    dim3 gridDims((unsigned int) ceil((double)(width*height/blockDims.x)), 1, 1 );
    blur<<<gridDims, blockDims>>>(dev_input, dev_output, width, height); 

    printf("Kernel finish\n");

    cudaMemcpy(output_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost );

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
    printf("Load image\n");
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
    filter(input_image, output_image, width, height); 



    // Prepare data for output
    // Add alpha channel in out image
    std::vector<unsigned char> out_image;
    for(int i = 0; i < width*height*3; i++) {
        out_image.push_back(output_image[i]);
        // printf("id: %d, output_image_val: %d\n", i, output_image[i]);
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
