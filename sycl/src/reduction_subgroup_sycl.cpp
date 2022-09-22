#include <sycl/sycl.hpp>
#include <iostream>
#include <stdio.h>
#include "time_ms.hpp"
#include <sycl_defines.hpp>


#include <utils.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#ifndef SIZE_REDUCTION
    #define SIZE_REDUCTION 30720
#endif
#define BLOCK_SIZE 512
#define N_BLOCKS (SIZE_REDUCTION/BLOCK_SIZE)
#define T float

namespace po = boost::program_options;

template<class X>
class sub_group_reduce{
    private:
            accessor<X, 1, access_mode::read> in;
            accessor<X, 1, access_mode::read_write> out;
            local_accessor<T,1> local_data;
    public:
        sub_group_reduce(
            accessor<X, 1, access_mode::read> in,
            accessor<X, 1, access_mode::read_write> out,
            local_accessor<X,1> local_data)
            :
            in(in),
            out(out),
            local_data(local_data)
            {}

            void operator()(nd_item<1> it) const{
                const auto &sub_group = it.get_sub_group();
                int local_id = it.get_local_id(0);
                int sub_group_id = sub_group.get_local_id();

                local_data[local_id] = in[it.get_global_id(0)];
                
                X mySum = 0;   
                mySum = reduce_over_group(sub_group, local_data[local_id], std::plus<X>());

                atomic_ref<X, memory_order::relaxed, memory_scope::device,access::address_space::global_space> ao(out[0]);
                if(sub_group_id==0)
                    ao.fetch_add(mySum);
            }

};

int main (int argc, char* argv[]){
    T* input = (T*)malloc(SIZE_REDUCTION * sizeof(T));
    T* output = (T*)malloc(sizeof(T));

    if (!input) // Check if malloc was all right
        return -1;

    for (int i = 0; i < SIZE_REDUCTION; i++)
        input[i] = 1.0f;

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
          gpu_platforms.push_back(p);
      }
      return gpu_platforms;
    }();
  
    auto device = gpu_platforms[0].get_devices()[0];

  
    queue Q{device, property::queue::enable_profiling()};
    {
        
        buffer<T, 1> in_buff{input, SIZE_REDUCTION};
        buffer<T, 1> out_buff{output, 1};
        #ifdef KERNEL_TIME
            // for each buf in buffers create a dummy kernel
            std::vector<buffer<T,1>> buffers;
            buffers.push_back(in_buff);
            buffers.push_back(out_buff);
            for(buffer<T,1> buf:buffers)
                utils::forceDataTransfer(Q, buf);
        #endif

        range<1> grid{BLOCK_SIZE*N_BLOCKS};
        range<1> block{BLOCK_SIZE};
        event e;
        

        e = Q.submit([&](handler &cgh){
            accessor in_acc{in_buff,cgh, read_only};
            accessor out_acc{out_buff,cgh, read_write};
            
            local_accessor<T,1> local_data{BLOCK_SIZE, cgh};

            cgh.parallel_for(nd_range<1>{grid, block}, sub_group_reduce<T>(
                in_acc,
                out_acc,
                local_data
            ));
        });
        time_ms(e, "reduction_subgroup_sycl");
    }

    #ifdef DEBUG
    if(*output==SIZE_REDUCTION)
        printf("Test PASS\n");
    else 
        printf("Test FAIL\n");
    #endif


}