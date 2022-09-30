
#include <stdio.h>
#include <iostream>
#include <time_ms.hpp>
#include <sycl_defines.hpp>
#include <utils.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


namespace po = boost::program_options;
#ifndef SIZE_REDUCTION
    #define SIZE_REDUCTION 30720
#endif
#define T float
int main(int argc, char* argv[]){
    // input data
    T *h_input = (T *) malloc(SIZE_REDUCTION * sizeof(T));
    // output
    T sumResult = 0;
    for(int i=0; i < SIZE_REDUCTION; i++)
        h_input[i]= 1;
    
    std::string use_sycl="";
   
    po::options_description desc("Allowed options");
    desc.add_options()
        ("sycl",po::value(&use_sycl), "use SYCL implementation with cpu or gpu")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    auto dev_type = utils::select_dev(use_sycl);
    auto platforms = sycl::platform::get_platforms();
    
    //Take all cpu or gpu platforms
    auto gpu_platforms = [&platforms, &dev_type](){
    std::vector<sycl::platform> gpu_platforms;
      for(auto& p : platforms){
        if(p.has(dev_type))
            if(p.get_info<sycl::info::platform::name>()==PLATFORM)
                 gpu_platforms.push_back(p);
      }
      return gpu_platforms;
    }();
  
    auto device = gpu_platforms[0].get_devices()[0];


    queue Q{device, property::queue::enable_profiling()};
    
    
    {
        // Buffers init
        event e;
        buffer<T, 1> inBuff {h_input, SIZE_REDUCTION};
        buffer<T, 1> sumBuf { &sumResult, 1 };
        
        #ifdef KERNEL_TIME
            // for each buf in buffers create a dummy kernel
            std::vector<buffer<T,1>> buffers;
            buffers.push_back(inBuff);
            buffers.push_back(sumBuf);
            for(buffer<T,1> buf:buffers)
                utils::forceDataTransfer(Q, buf);
        #endif

        e = Q.submit([&](handler& cgh) {

        // Input values to reductions are standard accessors
        auto inputValues = inBuff.get_access<access_mode::read>(cgh);
        auto sumValue = sumBuf.get_access<access_mode::read_write>(cgh);

        // Create temporary objects describing variables with reduction semantics
        auto sumReduction = reduction(sumValue,sycl::plus<T>());

        // parallel_for performs two reduction operations
        // For each reduction variable, the implementation:
        // - Creates a corresponding reducer
        // - Passes a reference to the reducer to the lambda as a parameter
        cgh.parallel_for(range<1>{SIZE_REDUCTION},
            sumReduction,
            [=](id<1> idx, auto& sum) {
            // plus<>() corresponds to += operator, so sum can be updated via += or combine()
            sum += inputValues[idx];
        });
        });

        time_ms(e, "reducer_sycl");

    }  
    #ifdef DEBUG
        if(sumResult==SIZE_REDUCTION){
            std::cout << "pass" << std::endl;
        }
        else
            std::cout <<  "fail" << std::endl;
    #endif
    return 0;

}