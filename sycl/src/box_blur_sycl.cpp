#include <iostream>
#include <cstdlib>
#include "../include/lodepng.h"
#include <functional>
#include <cmath>
#include <sycl/sycl.hpp>
#include "../include/time_ms.hpp"

#define BLOCK_SIZE 512
using namespace sycl;
class blur{
    private:
        const accessor<unsigned char, 1, access_mode::read> in;
        accessor<unsigned char, 1, access_mode::read_write> out;
        const int w;
        const int h;

    public:
        blur(
            const accessor<unsigned char, 1, access_mode::read>& in,
            accessor<unsigned char, 1, access_mode::read_write>& out,
            const int &w,
            const int &h
        ):
        in(in),
        out(out),
        w(w),
        h(h){}

        void operator()(nd_item<1> it) const {
            const auto&  group = it.get_group();
            int group_id = group.get_group_id();
            int local_id = group.get_local_id(0);

            const unsigned int offset = group_id * BLOCK_SIZE + local_id;
            int x = offset % w;
            int y = (offset-x)/w;
            int fsize = 5; // Filter size
            if(offset < w*h) {

                float output_red = 0;
                float output_green = 0;
                float output_blue = 0;
                int hits = 0;
                for(int ox = -fsize; ox < fsize+1; ++ox) {
                    for(int oy = -fsize; oy < fsize+1; ++oy) {
                        if((x+ox) > -1 && (x+ox) < w && (y+oy) > -1 && (y+oy) < h) {
                            const int currentoffset = (offset+ox+oy*w)*3;
                            output_red += in[currentoffset]; 
                            output_green += in[currentoffset+1];
                            output_blue += in[currentoffset+2];
                            hits++;
                        }
                    }
                }
                
                out[offset*3] = output_red/hits;
                out[offset*3+1] = output_green/hits;
                out[offset*3+2] = output_blue/hits;
            }
        }
};

void filter (unsigned char* input_image, unsigned char* output_image, int width, int height) {

    queue Q {gpu_selector(), property::queue::enable_profiling()};
    {
        range<1> threadPerBlock{BLOCK_SIZE};
        range<1> threadInGrid{BLOCK_SIZE*BLOCK_SIZE};
        buffer<unsigned char, 1> in_buff {input_image, width*height*3};
        buffer<unsigned char, 1> out_buff {output_image, width*height*3};
      
        event e;
        e = Q.submit([&](handler &cgh){
            const accessor<unsigned char, 1> in {in_buff, cgh, read_only};
            accessor<unsigned char, 1> out {out_buff, cgh, read_write, no_init};
            

            cgh.parallel_for(nd_range<1>{threadInGrid, threadPerBlock}, blur(
                    in,
                    out, 
                    width,
                    height
                ) //end blur class
            ); //end parallel for
            
        }); // end queue

        time_ms(e, "box_blur_sycl");
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
