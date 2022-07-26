#include <stdio.h>
#include <iostream>
#include <sycl/sycl.hpp>
#include "time_ms.hpp"

#define N 1024
#define BLOCK_SIZE 16


using namespace sycl;

//only square matrix with N multple of BLOCK_SIZE
template <int DIM_MATRIX>
class square_matrix_mul_tiling{
    private:
        const accessor<float, 1, sycl::access_mode::read, sycl::target::device> a_matrix;
        const accessor<float, 1, sycl::access_mode::read, sycl::target::device> b_matrix;
        accessor<float, 1, sycl::access_mode::read_write, sycl::target::device> c_matrix;
        local_accessor<float, 2> tile_a;
        local_accessor<float, 2> tile_b;

    public:
        square_matrix_mul_tiling(
                const accessor<float, 1, sycl::access_mode::read, sycl::target::device> a_matrix,
                const accessor<float, 1, sycl::access_mode::read, sycl::target::device> b_matrix,
                accessor<float, 1, sycl::access_mode::read_write, sycl::target::device> c_matrix,
                local_accessor<float, 2> tile_a,
                local_accessor<float, 2> tile_b
        )
        :
        a_matrix(a_matrix),
        b_matrix(b_matrix),
        c_matrix(c_matrix),
        tile_a(tile_a),
        tile_b(tile_b)
        {}

        void operator()(nd_item<2> it) const{
            const auto &group = it.get_group();
            int col = static_cast<int>(it.get_global_id(1));
            int row = static_cast<int>(it.get_global_id(0));

            int local_id_x = group.get_local_id(1);
            int local_id_y = group.get_local_id(0);
            
            float tmp = 0;
            int idx;

            int grid_dim = it.get_global_range(0) / BLOCK_SIZE;
         
            for (int sub = 0; sub < grid_dim; ++sub) 
            {
                // load data in local memory
                idx = row * DIM_MATRIX + sub * BLOCK_SIZE + local_id_x;
                
                tile_a[local_id_y][local_id_x] = a_matrix[idx];
                
                idx = (sub * BLOCK_SIZE + local_id_y) * DIM_MATRIX + col;
                
                tile_b[local_id_y][local_id_x] = b_matrix[idx];
                
                group_barrier(group);

                for (int k = 0; k < BLOCK_SIZE; ++k) 
                    tmp += tile_a[local_id_y][k] * tile_b[k][local_id_x];
    
                group_barrier(group);
            }
    
            c_matrix[row * DIM_MATRIX + col] = tmp;
        }
};


int main( int argc, char** argv)
{
    
    // Computation is divided into tiles of TILE_DIM X TILE_DIME (where TILE_DIM is multiple of BLOCK_ROWS). 
    // execution configuration parameters
 
    event e;
    
    // size of memory required to store the matrix
    const int mem_size = sizeof(float) * N*N;
    
   
    float *a_matrix = (float*) malloc(mem_size);
    float *b_matrix = (float*) malloc(mem_size);
    float *c_matrix = (float*) malloc(mem_size);

  
    for(int i = 0; i < (N*N); ++i){
        a_matrix[i] = (float) i+1;
        b_matrix[i] = (float) i+1;
        c_matrix[i] = 0.0;
    }

    queue Q{gpu_selector(), property::queue::enable_profiling()};

    {
        range<2> grid {N, N}; 
        range<2> block{BLOCK_SIZE, BLOCK_SIZE};
        buffer<float, 1> a_matrix_buff {a_matrix, N*N};
        buffer<float, 1> b_matrix_buff {b_matrix, N*N};
        buffer<float, 1> c_matrix_buff {c_matrix, N*N};


        e = Q.submit([&](handler &cgh){
            // input and output amtrix accessor
            const accessor<float, 1> a_matrix_acc {a_matrix_buff, cgh, read_only};
            const accessor<float, 1> b_matrix_acc {b_matrix_buff, cgh, read_only};
            accessor<float, 1> c_matrix_acc {c_matrix_buff, cgh, read_write};

            // local accessor for tile
            local_accessor<float, 2> tile_a{range<2>{BLOCK_SIZE, BLOCK_SIZE}, cgh};
            local_accessor<float, 2> tile_b{range<2>{BLOCK_SIZE, BLOCK_SIZE}, cgh};

            cgh.parallel_for(nd_range<2>{grid, block}, square_matrix_mul_tiling<N>(
                a_matrix_acc,
                b_matrix_acc,
                c_matrix_acc,
                tile_a,
                tile_b
            )
            );//end parallel for
        });
        time_ms(e, "matrix_mul_tiling_sycl");
    }
    #ifdef DEBUG
    for(int i = 0; i < N*N; i++)
        std::cout << c_matrix[i] << ", ";
    #endif
    return 0;
}