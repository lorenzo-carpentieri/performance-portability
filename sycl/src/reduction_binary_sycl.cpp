/*
    Baseline
*/
#include <sycl/sycl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "time_ms.hpp"


using namespace sycl;
#define T float

#define BLOCK_SIZE 512
#define N 30720
#define N_BLOCKS (N/BLOCK_SIZE)


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

        void operator()(nd_item<1> it) const{
            const auto &group = it.get_group();
            int grid_size = N_BLOCKS * BLOCK_SIZE;
            T my_sum = 0;

            unsigned int idx = group.get_local_id(0);

            // In this case we also halve the number of blocks
            // threads in block0 sum data in block_0 and block_1
            // threads in block_1 sum data in block_2 and block_3
            // ...
            // starting index 
            unsigned int i = group.get_group_id(0) * BLOCK_SIZE * 2 + idx;
            grid_size = grid_size << 1;
            while (i < N) {
                my_sum += input[i];
                
                if ((i + BLOCK_SIZE) < N)
                    my_sum += input[i + BLOCK_SIZE];

                i += grid_size;
            }
        
            // each thread puts its local sum into shared memory
            local_data[idx] = my_sum;
            group_barrier(group);
           
            //TODO: 32 dipende dall work_group sostituire facendo la chiamata per capire la sub_group supportat
            for (unsigned int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1) {
                if (idx < stride) // Only the first half of the threads do the computation
                    local_data[idx] += local_data[idx + stride];

                group_barrier(group); // Wait that all threads in the block compute the partial sum
            }

            // TODO: 32 non va bene per tutti 
            if(idx < 32){
                if (BLOCK_SIZE >= 64) local_data[idx] += local_data[idx + 32];
                if (BLOCK_SIZE >= 32) local_data[idx] += local_data[idx + 16];
                if (BLOCK_SIZE >= 16) local_data[idx] += local_data[idx + 8];
                if (BLOCK_SIZE >= 8) local_data[idx] += local_data[idx + 4];
                if (BLOCK_SIZE >= 4) local_data[idx] += local_data[idx + 2];
                if (BLOCK_SIZE >= 2) local_data[idx] += local_data[idx + 1];
            }
          
            atomic_ref<T, memory_order::relaxed, memory_scope::device,access::address_space::global_space> ao(output[0]);
            if (idx == 0) {
                ao.fetch_add(local_data[0]);
            }

        }       
};


int main()
{
    
    T *h_input = (T *) malloc(N * sizeof(T));
    T *h_output = (T *) malloc(sizeof(T));

    if (!h_input) // Check if malloc was all right
        return -1;

    for (int i = 0; i < N; i++)
        h_input[i] = 1.0f;
    

    queue Q{gpu_selector(), property::queue::enable_profiling()};
 

  
    
    {
        // Init buffer
        buffer<T,1> in_buff {h_input, N};
        buffer<T, 1> out_buff {h_output, 1};

        // event
        event e;
        e = Q.submit([&](handler &cgh){
                
                accessor<T,1> in_acc {in_buff, cgh, read_only}; 
                accessor<T,1> out_acc {out_buff, cgh, read_write}; 
                // shared memory
                local_accessor<T, 1> local_acc{range<1>{BLOCK_SIZE}, cgh};

                cgh.parallel_for(
                    nd_range<1>{N, BLOCK_SIZE},
                    binary_reduction(
                        in_acc,
                        out_acc,
                        local_acc
                    )
                );
            }
        ); //end submit
        time_ms(e, "reduction_binary_sycl");

    }

    
    // #ifdef DEBUG
    if (*h_output == N)
        printf("pass\n");
    else
        printf("fail, result: %f\n", *h_output);
    // #endif

    return 0;
}