#include <stdio.h>
#include <iostream>
#include <sycl/sycl.hpp>
// kernels transpose/copy a tile of TILE_DIM x TILE_DIM elements
// using a TILE_DIM x BLOCK_ROWS thread block, so that each thread
// transposes TILE_DIM/BLOCK_ROWS elements. TILE_DIM must be an
// integral multiple of BLOCK_ROWS
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define SIZE_X 2048
#define SIZE_Y 2048

// Number of repetitions used for timing.
#define NUM_REPS 100

using namespace sycl;

// width, height matrix dimensions
template <int DIM_X, int DIM_Y, int TILE_SIZE>
class transposeCoalesced{
    private:
        const accessor<float, 1, sycl::access_mode::read, sycl::target::device> in_matrix;
        accessor<float, 1, sycl::access_mode::read_write, sycl::target::device> out_matrix;
        sycl::local_accessor<float, 2> tile;
   

    public:
        transposeCoalesced(
                const accessor<float, 1, sycl::access_mode::read, sycl::target::device> in_matrix,
                accessor<float, 1, sycl::access_mode::read_write, sycl::target::device> out_matrix,
                sycl::local_accessor<float, 2> tile
        )
        :
        in_matrix(in_matrix),
        out_matrix(out_matrix),
        tile(tile){}

        void operator()(nd_item<2> it) const{
            const auto group = it.get_group();
            int local_id_x=it.get_local_id(1);
            int local_id_y=it.get_local_id(0);

            int block_x = group.get_group_id(1);
            int block_y = group.get_group_id(0);

            int xIndex = block_x * TILE_SIZE +local_id_x;
            int yIndex = block_y * TILE_SIZE + local_id_y;
            
            int index_in = xIndex + (yIndex)*DIM_Y;

            xIndex = block_y * TILE_SIZE + local_id_x;
            yIndex = block_x * TILE_SIZE + local_id_y;
            int index_out = xIndex + (yIndex) * DIM_X;
            
            // Copy data in local memory
            for (int i=0; i<TILE_SIZE; i+=BLOCK_ROWS) {
                tile[local_id_y+i][local_id_x] = in_matrix[index_in+i*DIM_Y];
            }
            
            group_barrier(group);

            for (int i=0; i<TILE_SIZE; i+=BLOCK_ROWS) {
                out_matrix[index_out+i*DIM_X] = tile[local_id_x][local_id_y+i];
            }
        }


};
// (float *odata, float *idata, int width, int DIM_X)
// {
    // initialize local memory
    // TILE_DIM+1 to avoid bank conflicts 

// }


int main( int argc, char** argv)
{
    
    // Computation is divided into tiles of TILE_DIM X TILE_DIME (where TILE_DIM is multiple of BLOCK_ROWS). 
    // execution configuration parameters
    
    // CUDA events
    event e;
    
    // size of memory required to store the matrix
    const int mem_size = sizeof(float) * SIZE_X*SIZE_Y;
    
   
    float *in_matrix = (float*) malloc(mem_size);
    float *out_matrix = (float*) malloc(mem_size);
  
    for(int i = 0; i < (SIZE_X*SIZE_Y); ++i)
        in_matrix[i] = (float) i;

    queue Q;

    {
        range<2> grid {BLOCK_ROWS * (SIZE_X / TILE_DIM), TILE_DIM * (SIZE_X / TILE_DIM)}; 
        range<2> block{BLOCK_ROWS, TILE_DIM};
        buffer<float, 1> in_matrix_buff {in_matrix, SIZE_X*SIZE_Y};
        buffer<float, 1> out_matrix_buff {out_matrix, SIZE_X*SIZE_Y};
        e = Q.submit([&](handler &cgh){
            // input and output amtrix accessor
            const accessor<float, 1> in_matrix_acc {in_matrix_buff, cgh, read_only};
            accessor<float, 1> out_matrix_acc {out_matrix_buff, cgh, read_write};

            // local memory with TILE_DIM+1 to avoid bank conflicts
            local_accessor<float,2> tile {range<2>{TILE_DIM, TILE_DIM+1}, cgh};
            
            cgh.parallel_for(nd_range<2>{grid, block}, transposeCoalesced<SIZE_X, SIZE_Y, TILE_DIM>(
                in_matrix_acc,
                out_matrix_acc,
                tile
            )
            );//end parallel for
        });

    }

    for(int i = 0; i < SIZE_X*SIZE_Y; i++)
        std::cout << out_matrix[i] << ", ";


    
    
    return 0;
}