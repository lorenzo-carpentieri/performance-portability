#include <iostream>
#include <cstdlib>
#include "../include/lodepng.h"
#include <functional>
#include <cmath>
#include <sycl/sycl.hpp>
#include "../include/time_ms.hpp"

#ifndef RADIUS
    #define RADIUS         3
#endif
#define D              (RADIUS*2+1)
#define S              (D*D)
#define BLOCK_SIZE     512
#define IMG_SIZE       512

using namespace sycl;
class blur{
    private:
        const accessor<unsigned char, 1, access_mode::read> in_r;
        const accessor<unsigned char, 1, access_mode::read> in_g;
        const accessor<unsigned char, 1, access_mode::read> in_b;

        accessor<unsigned char, 1, access_mode::read_write> out;

        local_accessor<unsigned char, 2> smem_r;
        local_accessor<unsigned char, 2> smem_g;
        local_accessor<unsigned char, 2> smem_b;


        const int w;
        const int h;

    public:
        blur(
            const accessor<unsigned char, 1, access_mode::read>& in_r,
            const accessor<unsigned char, 1, access_mode::read>& in_g,
            const accessor<unsigned char, 1, access_mode::read>& in_b,


            accessor<unsigned char, 1, access_mode::read_write>& out,
            local_accessor<unsigned char, 2> &smem_r,
            local_accessor<unsigned char, 2> &smem_g,
            local_accessor<unsigned char, 2> &smem_b,
            const int &w,
            const int &h
        ):
        in_r(in_r),
        in_g(in_g),
        in_b(in_b),
        out(out),
        smem_r(smem_r),
        smem_g(smem_g),
        smem_b(smem_b),
        w(w),
        h(h){}

        void operator()(nd_item<1> it) const {
            const auto&  group = it.get_group();
            int group_id = group.get_group_id();
            int local_id = group.get_local_id(0);
            const int NUM_BLOCKS = w*h / IMG_SIZE;

            const int gidx = group_id * BLOCK_SIZE + local_id;
            const int x = gidx % w;
            const int y = (gidx-x)/w;
        
            int row = 0;
            
            //load data in shared memory: each thread load in shared memory the upper and lower row 
            for(int i = -RADIUS; i < RADIUS + 1; i++){
                    if((static_cast<int>(group_id) + i) > -1 && (static_cast<int>(group_id) + i) < NUM_BLOCKS){
                        smem_r[row][local_id] = in_r[gidx+(i*w)];
                        smem_g[row][local_id] = in_g[gidx+(i*w)];
                        smem_b[row][local_id] = in_b[gidx+(i*w)];
                    }
                    else{
                        smem_r[row][local_id] = 0;
                        smem_g[row][local_id] = 0;
                        smem_b[row][local_id] = 0;
                    }
                    row++;
            }
            group_barrier(group);
            
                // box filter (only for threads inside the tile)
                
                float sum_r = 0;
                float sum_g = 0;
                float sum_b = 0;
                int hits = 0;
            
                for(int ox = -RADIUS; ox < RADIUS+1; ++ox) {
                    for(int oy = -RADIUS; oy < RADIUS+1; ++oy) {
                        if((x+ox) > -1 && (x+ox) < w && (y+oy) > -1 && (y+oy) < h) {
                            sum_r += smem_r[RADIUS+ox][x+oy]; 
                            sum_g += smem_g[RADIUS+ox][x+oy];
                            sum_b += smem_b[RADIUS+ox][x+oy];
                            hits++;
                        }
                    }
                }
                out[gidx*3  ] = sum_r / hits;
                out[gidx*3+1] = sum_g / hits;       
                out[gidx*3+2] = sum_b / hits;        
        }
};

void filter (unsigned char* input_r,unsigned char* input_g,unsigned char* input_b, unsigned char* output_image, const int width, const int height) {

    queue Q {gpu_selector(), property::queue::enable_profiling()};
    {
        range<1> threadPerBlock{BLOCK_SIZE};
        range<1> threadInGrid{BLOCK_SIZE*BLOCK_SIZE};
        buffer<unsigned char, 1> in_r_buff {input_r, width*height};
        buffer<unsigned char, 1> in_g_buff {input_g, width*height};
        buffer<unsigned char, 1> in_b_buff {input_b, width*height};

        buffer<unsigned char, 1> out_buff {output_image, width*height*3};
        event e;

        e=Q.submit([&](handler &cgh){
            // input accessors(red, green, blue)
            const accessor<unsigned char, 1> in_r {in_r_buff, cgh, read_only};
            const accessor<unsigned char, 1> in_g {in_g_buff, cgh, read_only};
            const accessor<unsigned char, 1> in_b {in_b_buff, cgh, read_only};
            
            // output accessor
            accessor<unsigned char, 1> out {out_buff, cgh, read_write, no_init};
            
            // local memory
            local_accessor<unsigned char, 2> smem_r {range<2>{D, BLOCK_SIZE}, cgh};
            local_accessor<unsigned char, 2> smem_g {range<2>{D, BLOCK_SIZE}, cgh};
            local_accessor<unsigned char, 2> smem_b {range<2>{D, BLOCK_SIZE}, cgh};


            

            cgh.parallel_for(nd_range<1>{threadInGrid, threadPerBlock}, blur(
                    in_r,
                    in_g,
                    in_b,
                    out,
                    smem_r,
                    smem_g,
                    smem_b,
                    width,
                    height
                ) //end blur class
            ); //end parallel for
            
        }); // end queue

        time_ms(e, "box_blur_local_memory_sycl");
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
    for(int i = 0; i < static_cast<int>(in_image.size()); i++) {
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


    for(uint i = 0; i < width * height; i++){
        input_r[i] = input_image[i*3];
        input_g[i] = input_image[i*3+1];
        input_b[i] = input_image[i*3+2];
    }
    


    // Run the filter on it
    filter(input_r, input_g, input_b, output_image, width, height); 
 



    // Prepare data for output
    // Add alpha channel in out image
    std::vector<unsigned char> out_image;
    for(uint i = 0; i < width*height*3; i++) {
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
