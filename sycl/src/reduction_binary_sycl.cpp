/*
    Baseline
*/
#include <stdio.h>
#include "time_ms.hpp"
#include <sycl_defines.hpp>
#include <utils.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>



#define T float

#define BLOCK_SIZE 512
#ifndef SIZE_REDUCTION
    #define SIZE_REDUCTION 30720
#endif
#define N_BLOCKS (SIZE_REDUCTION/BLOCK_SIZE)

namespace po = boost::program_options;

template<int SUB_GROUP_DIM>
class binary_reduction{
    private:
        accessor<T,1, access_mode::read> input;
        accessor<T,1, access_mode::read_write> output;
        local_accessor<T,1> local_data;
        


    public:
        
        binary_reduction(
            accessor<T,1, access_mode::read> input,
            accessor<T,1, access_mode::read_write> output,
            local_accessor<T,1> local_data
        ):
        input(input),
        output(output),
        local_data(local_data)
        {}

        void operator()(nd_item<1> it) const {
            const auto &group = it.get_group();
            const auto &sub_group = it.get_sub_group();

            int grid_size = N_BLOCKS * BLOCK_SIZE;
            T my_sum = 0;
           
            unsigned int idx = it.get_local_id(0);
            
            // // In this case we also halve the number of blocks
            // // threads in block0 sum data in block_0 and block_1
            // // threads in block_1 sum data in block_2 and block_3
            // // ...
            // // starting index 
            int i = it.get_group(0) * BLOCK_SIZE * 2 + idx;
            grid_size = grid_size << 1;
            while (i < SIZE_REDUCTION) {
                my_sum += input[i];
                
                if ((i + BLOCK_SIZE) < SIZE_REDUCTION)
                    my_sum += input[i + BLOCK_SIZE];

                i += grid_size;
            }
    
            // // each thread puts its local sum into shared memory
            local_data[idx] = my_sum;
            group_barrier(group);

            for (unsigned int stride = BLOCK_SIZE / 2; stride > SUB_GROUP_DIM; stride >>= 1) {
                if (idx < stride) // Only the first half of the threads do the computation
                    local_data[idx] += local_data[idx + stride];
                group_barrier(group); // Wait that all threads in the block compute the partial sum
            }

            if(idx < SUB_GROUP_DIM){
                if(SUB_GROUP_DIM == 1){
                    if (BLOCK_SIZE >= 2) local_data[idx] += local_data[idx + 1];
                }
                else if(SUB_GROUP_DIM == 16){
                    if (BLOCK_SIZE >= 32) local_data[idx] += local_data[idx+ 16];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 16) local_data[idx] += local_data[idx+ 8];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 8) local_data[idx] += local_data[idx + 4];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 4) local_data[idx] += local_data[idx + 2];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 2) local_data[idx] += local_data[idx + 1];
                } 
                else if(SUB_GROUP_DIM == 32){
                    if (BLOCK_SIZE >= 64) local_data[idx] += local_data[idx+ 32];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 32) local_data[idx] += local_data[idx+ 16];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 16) local_data[idx] += local_data[idx+ 8];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 8) local_data[idx] += local_data[idx + 4];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 4) local_data[idx] += local_data[idx + 2];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 2) local_data[idx] += local_data[idx + 1];
                } 
                else if(SUB_GROUP_DIM==64){
                    if (BLOCK_SIZE >= 128) local_data[idx] += local_data[idx+ 64];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 64) local_data[idx] += local_data[idx+ 32];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 32) local_data[idx] += local_data[idx+ 16];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 16) local_data[idx] += local_data[idx+ 8];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 8) local_data[idx] += local_data[idx + 4];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 4) local_data[idx] += local_data[idx + 2];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 2) local_data[idx] += local_data[idx + 1];
                }   

                else{
                    if (BLOCK_SIZE >= 256) local_data[idx] += local_data[idx+ 128];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 128) local_data[idx] += local_data[idx+ 64];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 64) local_data[idx] += local_data[idx+ 32];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 32) local_data[idx] += local_data[idx+ 16];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 16) local_data[idx] += local_data[idx+ 8];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 8) local_data[idx] += local_data[idx + 4];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 4) local_data[idx] += local_data[idx + 2];
                    group_barrier(sub_group);
                    if (BLOCK_SIZE >= 2) local_data[idx] += local_data[idx + 1];
                }           
             }
                
        

            atomic_ref<T, memory_order::relaxed, memory_scope::device,access::address_space::global_space> ao(output[0]);
            if (idx == 0) {
                ao.fetch_add(local_data[0]);
            }

        }       
};


int main(int argc, char* argv[])
{
    
    T *h_input = (T *) malloc(SIZE_REDUCTION * sizeof(T));
    T *h_output = (T *) malloc(sizeof(T));

    if (!h_input) // Check if malloc was all right
        return -1;

    for (int i = 0; i < SIZE_REDUCTION; i++)
        h_input[i] = 1.0f;
    
    std::string use_sycl="";
    po::options_description desc("Allowed options");
    desc.add_options()
        ("sycl",po::value(&use_sycl), "use SYCL implementation with cpu or gpu");

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
        // Init buffer
        buffer<T,1> in_buff {h_input, SIZE_REDUCTION};
        buffer<T, 1> out_buff {h_output, 1};
        #ifdef KERNEL_TIME
            // for each buf in buffers create a dummy kernel
            std::vector<buffer<T,1>> buffers;
            buffers.push_back(in_buff);
            buffers.push_back(out_buff);
            for(buffer<T,1> buf:buffers)
                utils::forceDataTransfer(Q, buf);
        #endif

        // Different devices can use different sub group size
	    const auto subgroup_size = Q.get_device().get_info<sycl::info::device::sub_group_sizes>().at(0);

        // event
        event e;
        e = Q.submit([&](handler &cgh){
                
                accessor in_acc {in_buff, cgh, read_only}; 
                accessor out_acc {out_buff, cgh, read_write}; 
                // shared memory
                local_accessor<T, 1> local_acc{BLOCK_SIZE, cgh};
                // luanch the kernel with the correct subgroup size
                switch(subgroup_size){
                    case 1: 
                        cgh.parallel_for(
                            nd_range<1>{range<1>{SIZE_REDUCTION}, range<1>{BLOCK_SIZE}},
                            binary_reduction<1>(
                                in_acc,
                                out_acc,
                                local_acc
                        ));
                        break;
                    case 16:
                        cgh.parallel_for(
                            nd_range<1>{range<1>{SIZE_REDUCTION}, range<1>{BLOCK_SIZE}},
                            binary_reduction<16>(
                                in_acc,
                                out_acc,
                                local_acc
                        ));
                        break;
                    case 32:
                        cgh.parallel_for(
                            nd_range<1>{range<1>{SIZE_REDUCTION}, range<1>{BLOCK_SIZE}},
                            binary_reduction<32>(
                                in_acc,
                                out_acc,
                                local_acc
                        ));   
                        break;
                    case 64:
                        cgh.parallel_for(
                            nd_range<1>{range<1>{SIZE_REDUCTION}, range<1>{BLOCK_SIZE}},
                            binary_reduction<64>(
                                in_acc,
                                out_acc,
                                local_acc
                        ));
                        break;
                    case 128:
                        cgh.parallel_for(
                            nd_range<1>{range<1>{SIZE_REDUCTION}, range<1>{BLOCK_SIZE}},
                            binary_reduction<128>(
                                in_acc,
                                out_acc,
                                local_acc
                        ));
                        break;
                    default:
                        throw std::runtime_error("Unsupported subgroup size");
                }

                
            }
        ); //end submit
        time_ms(e, "reduction_binary_sycl");

    }

    
    #ifdef DEBUG
    if (*h_output == SIZE_REDUCTION)
        printf("pass\n");
    else
        printf("fail, result: %f\n", *h_output);
    #endif

    // return 0;
}