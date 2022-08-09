#include <sycl/sycl.hpp>
#include <iostream>
#include <stdio.h>
#include "time_ms.hpp"

#define N 30720
#define BLOCK_SIZE 512
#define N_BLOCKS (N/BLOCK_SIZE)
#define T float

using namespace sycl;
template<class X>
class reduce{
    private:
            accessor<X, 1, access_mode::read> in;
            accessor<X, 1, access_mode::read_write> out;
            local_accessor<T,1> local_data;
    public:
        reduce(
            accessor<X, 1, access_mode::read> in,
            accessor<X, 1, access_mode::read_write> out,
            local_accessor<X,1> local_data)
            :
            in(in),
            out(out),
            local_data(local_data)
            {}

            void operator()(nd_item<1> it) const{
                const auto &group = it.get_group();
                const auto &sub_group = it.get_sub_group();
                int local_id = group.get_local_id(0);
                int sub_group_id = sub_group.get_local_id();

                local_data[local_id] = in[it.get_global_id(0)];
                
                X mySum = 0;   
                mySum = reduce_over_group(sub_group, local_data[local_id], std::plus<X>());

                atomic_ref<X, memory_order::relaxed, memory_scope::device,access::address_space::global_space> ao(out[0]);
                if(sub_group_id==0)
                    ao.fetch_add(mySum);
            }

};

int main (){
    T* input = (T*)malloc(N * sizeof(T));
    T* output = (T*)malloc(sizeof(T));

    int n_iterations = 200;

    if (!input) // Check if malloc was all right
        return -1;

    for (int i = 0; i < N; i++)
        input[i] = 1.0f;

    queue Q{gpu_selector(), property::queue::enable_profiling()};
    {
        buffer<T, 1> in_buff{input, N};
        buffer<T, 1> out_buff{output, 1};
        range<1> grid{BLOCK_SIZE*N_BLOCKS};
        range<1> block{BLOCK_SIZE};
        event e;
        e = Q.submit([&](handler &cgh){
            accessor<T, 1> in_acc{in_buff,cgh, read_only};
            accessor<T, 1> out_acc{out_buff,cgh, read_write};
            
            local_accessor<T,1> local_data{BLOCK_SIZE, cgh};

            cgh.parallel_for(nd_range<1>{grid, block}, reduce<T>(
                in_acc,
                out_acc,
                local_data
            ));
        });
        time_ms(e, "reduction_sycl");
    }

    #ifdef DEBUG
    if(*output==N)
        printf("Test PASS\n");
    else 
        printf("Test FAIL\n");
    #endif


}