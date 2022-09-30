#include <iostream>
#include <cstdlib>
#include "../include/lodepng.h"
#include <functional>
#include <cmath>
#include "../include/time_ms.hpp"
#include <sycl_defines.hpp>
#include <utils.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


namespace po = boost::program_options;

#define BLOCK_SIZE 512
#ifndef RADIUS_SOBEL
    #define RADIUS_SOBEL 3
#endif
using namespace sycl;

class sobel_filter_sycl{
    private:
        const uint size;
        const accessor<sycl::float3, 1, access_mode::read, target::device> in;
        accessor<sycl::float3, 1, access_mode::read_write, target::device> out;
        const accessor<float, 1, access_mode::read, target::device> kernel;       

    public:
        sobel_filter_sycl(
            const uint &size,
            const accessor<sycl::float3, 1, access_mode::read> in,
            accessor<sycl::float3, 1, access_mode::read_write> out,
            const accessor<float, 1, access_mode::read> kernel
            
        ):
        size(size),
        in(in),
        out(out),
        kernel(kernel)
        {}

        void operator()(id<2> gid) const {
            uint x = gid[0];
            uint y = gid[1];
            
           
            sycl::float3 Gx = sycl::float3(0, 0, 0);
            sycl::float3 Gy = sycl::float3(0, 0, 0);
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
                    int offset_x = x_shift + y_shift * RADIUS_SOBEL;
                    int offset_y = y_shift + x_shift * RADIUS_SOBEL;

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

void filter (sycl::float3* input_image, sycl::float3* output_image, int width, int height, std::string use_sycl) {
    auto dev_type = utils::select_dev(use_sycl);
    auto platforms = sycl::platform::get_platforms();
    
    //Take all cpu or gpu platforms
auto gpu_platforms = [&platforms, &dev_type](){
  std::vector<sycl::platform> gpu_platforms;
  for(auto& p : platforms){
    std::string platform_name;
    if(p.has(dev_type))
      platform_name = p.get_info<sycl::info::platform::name>();
      if(platform_name.find(PLATFORM)!=-1)
        gpu_platforms.push_back(p);
  }
      return gpu_platforms;
  }();
  
    auto device = gpu_platforms[0].get_devices()[0];

    queue Q {device, property::queue::enable_profiling()};
    {

        const float kernel [] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
        buffer<float, 1> kernel_buff {kernel, 9};
        buffer<sycl::float3, 1> in_buff {input_image, width*height};
        buffer<sycl::float3, 1> out_buff {output_image, width*height};

        #ifdef KERNEL_TIME
            // for each buf in buffers create a dummy kernel
            std::vector<buffer<sycl::float3,1>> buffers;
            buffers.push_back(in_buff);
            buffers.push_back(out_buff);
            for(buffer<sycl::float3,1> buf:buffers)
                utils::forceDataTransfer(Q, buf);
            utils::forceDataTransfer(Q, kernel_buff);
        #endif
        // input image is N x N
        const uint &size = width;
        event e;
        range<2> range{size, size};


        e = Q.submit([&](handler &cgh){
            const accessor in {in_buff, cgh, read_only};
            const accessor  kernel_acc {kernel_buff, cgh, read_only};
            accessor out {out_buff, cgh, read_write, no_init};


            cgh.parallel_for(range, sobel_filter_sycl(
                    size,
                    in,
                    out, 
                    kernel_acc
                ) //end blur class
            ); //end parallel for
            
        }); // end queue

        time_ms(e, "sobel_filter_sycl");
    }
}

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cout << "Run with input and output image filenames." << std::endl;
        return 0;
    }
    std::string use_sycl="";
    po::options_description desc("Allowed options");
    desc.add_options()
        ("sycl",po::value(&use_sycl), "use SYCL implementation with cpu or gpu")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 
    // Read the arguments
    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::vector<unsigned char> in_image;
    unsigned int width, height;

    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    // Prepare the data
    unsigned char* input_image = new unsigned char[width*height*3];
    unsigned char* output_image = new unsigned char[width*height*3];

    sycl::float3* output_rgb = (sycl::float3*) malloc(sizeof(sycl::float3)* width*height);


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
    
    // unsigned char* input_r = new unsigned char[width*height];
    // unsigned char* input_g = new unsigned char[width*height];
    // unsigned char* input_b = new unsigned char[width*height];

    sycl::float3* input_rgb = (sycl::float3*) malloc(sizeof(sycl::float3)* width*height);
    

    for(uint i = 0; i < width * height; i++)
        input_rgb[i] = {input_image[i*3]/255.f, input_image[i*3+1]/255.f, input_image[i*3+2]/255.f};

    
    // Run the filter on it
    filter(input_rgb, output_rgb, width, height, use_sycl);

    for(uint i = 0; i < width * height; i++){
        int j = i*3;
        output_image[j] = output_rgb[i].x()*255.f;
        output_image[j+1] = output_rgb[i].y()*255.f;
        output_image[j+2] = output_rgb[i].z()*255.f;
    }


 

    // Prepare data for output
    // Add alpha channel in out image
    std::vector<unsigned char> out_image;
    for(uint i = 0; i < width*height*3; i++) {
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
