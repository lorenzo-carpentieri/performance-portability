#include <iostream>
#include <cstdlib>
#include "../include/lodepng.h"
#include <functional>
#include <cmath>
#include <sycl/sycl.hpp>
#include "../include/time_ms.hpp"

#define BLOCK_SIZE 512
using namespace sycl;

class sobel_filter{
    private:
        const accessor<sycl::float3, 1, access_mode::read> in;
        const accessor<float, 1, access_mode::read> kernel;
        accessor<sycl::float3, 1, access_mode::read_write> out;

        const int size;

    public:
        sobel_filter(
            const accessor<sycl::float3, 1, access_mode::read>& in,
            accessor<sycl::float3, 1, access_mode::read_write>& out,
            const int &size,
            const accessor<float, 1, access_mode::read> kernel
            
        ):
        in(in),
        out(out),
        size(size),
        kernel(kernel)
        {}

        void operator()(id<2> gid) const {
            int x = gid[0];
            int y = gid[1];
            
           
            sycl::float3 Gx = sycl::float3(0, 0, 0);
            sycl::float3 Gy = sycl::float3(0, 0, 0);
            const int radius = 3;
            // constant-size loops in [0,1,2]
            for(int x_shift = 0; x_shift < 3; x_shift++) {
                for(int y_shift = 0; y_shift < 3; y_shift++) {
                    // sample position
                    uint xs = x + x_shift - 1; // [x-1,x,x+1]
                    uint ys = y + y_shift - 1; // [y-1,y,y+1]
                    // for the same pixel, convolution is always 0
                    if(x == xs && y == ys)
                    continue;
                    // boundary check
                    if(xs < 0 || xs >= size || ys < 0 || ys >= size)
                    continue;

                    // sample color
                    sycl::float3 sample = in[xs*size+ys];
                    // convolution calculation
                    int offset_x = x_shift + y_shift * radius;
                    int offset_y = y_shift + x_shift * radius;

                    float conv_x = kernel[offset_x];
                    sycl::float3 conv4_x = sycl::float3(conv_x);
                    Gx += conv4_x * sample;

                    float conv_y = kernel[offset_y];
                    sycl::float3 conv4_y = sycl::float3(conv_y);


                    Gy += conv4_y * sample;
                }
            }
            
            // taking root of sums of squares of Gx and Gy
            sycl::float3 color = hypot(Gx, Gy);
            sycl::float3 minval = sycl::float3(0.0, 0.0, 0.0);
            sycl::float3 maxval = sycl::float3(1.0, 1.0, 1.0);
            
        
            out[gid.get(0)*size+gid.get(1)] = clamp(color, minval, maxval); 
        }
};

void filter (sycl::float3* input_image, sycl::float3* output_image, int width, int height) {

    queue Q {gpu_selector(), property::queue::enable_profiling()};
    {

        const float kernel [] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
        buffer<float, 1> kernel_buff {kernel, 9};
        buffer<sycl::float3, 1> in_buff {input_image, width*height};
        buffer<sycl::float3, 1> out_buff {output_image, width*height};
        // input image is N x N
        const size_t size = width;
        event e;
        range<2> range{size, size};


        e = Q.submit([&](handler &cgh){
            const accessor<sycl::float3, 1> in {in_buff, cgh, read_only};
            const accessor<float,1> kernel_acc {kernel_buff, cgh, read_only};
            accessor<sycl::float3, 1> out {out_buff, cgh, read_write, no_init};


            cgh.parallel_for(range, sobel_filter(
                    in,
                    out, 
                    size,
                    kernel_acc
                ) //end blur class
            ); //end parallel for
            
        }); // end queue

        time_ms(e, "sobel_filter");
    }
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

    sycl::float3* output_rgb = (sycl::float3*) malloc(sizeof(sycl::float3)* width*height);


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

    sycl::float3* input_rgb = (sycl::float3*) malloc(sizeof(sycl::float3)* width*height);
    

    for(int i = 0; i < width * height; i++)
        input_rgb[i] = {input_image[i*3]/255.f, input_image[i*3+1]/255.f, input_image[i*3+2]/255.f};

    
    // Run the filter on it
    filter(input_rgb, output_rgb, width, height);

    for(int i = 0; i < width * height; i++){
        int j = i*3;
        output_image[j] = output_rgb[i].x()*255.f;
        output_image[j+1] = output_rgb[i].y()*255.f;
        output_image[j+2] = output_rgb[i].z()*255.f;
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
