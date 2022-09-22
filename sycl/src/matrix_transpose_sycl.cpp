#include <stdio.h>
#include <iostream>

#include <time_ms.hpp>
#include <sycl_defines.hpp>

#include <utils.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


namespace po = boost::program_options;
// kernels transpose/copy a tile of TILE_DIM x TILE_DIM elements
// using a TILE_DIM x BLOCK_ROWS thread block, so that each thread
// transposes TILE_DIM/BLOCK_ROWS elements. TILE_DIM must be an
// integral multiple of BLOCK_ROWS

#define TILE_DIM 32
#define BLOCK_ROWS 8

#ifndef SIZE_X
    #define SIZE_X 2048
#endif
#ifndef SIZE_Y
    #define SIZE_Y 2048
#endif


// width, height matrix dimensions
template <int DIM_X, int DIM_Y, int TILE_SIZE>
class matrix_transpose{
    private:
        const accessor<float, 1, access_mode::read, target::device> in_matrix;
        accessor<float, 1, access_mode::read_write, target::device> out_matrix;
        local_accessor<float, 2> tile;
   

    public:
        matrix_transpose(
                const accessor<float, 1, access_mode::read, target::device> in_matrix,
                accessor<float, 1, access_mode::read_write, target::device> out_matrix,
                local_accessor<float, 2> tile
        )
        :
        in_matrix(in_matrix),
        out_matrix(out_matrix),
        tile(tile){}

        void operator()(nd_item<2> it) const{
            const auto group = it.get_group();
            int local_id_x=it.get_local_id(1);
            int local_id_y=it.get_local_id(0);

            int block_x = it.get_group(1);
            int block_y = it.get_group(0);

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


int main(int argc, char* argv[])
{
    
    // Computation is divided into tiles of TILE_DIM X TILE_DIME (where TILE_DIM is multiple of BLOCK_ROWS). 
    // execution configuration parameters
    
    event e;
    
    // size of memory required to store the matrix
    const int mem_size = sizeof(float) * SIZE_X*SIZE_Y;
    
   
    float *in_matrix = (float*) malloc(mem_size);
    float *out_matrix = (float*) malloc(mem_size);
  
    for(int i = 0; i < (SIZE_X*SIZE_Y); ++i)
        in_matrix[i] = (float) i;

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
        event e;
        range<2> grid {BLOCK_ROWS * (SIZE_X / TILE_DIM), TILE_DIM * (SIZE_X / TILE_DIM)}; 
        range<2> block{BLOCK_ROWS, TILE_DIM};
        buffer<float, 1> in_matrix_buff {in_matrix, SIZE_X*SIZE_Y};
        buffer<float, 1> out_matrix_buff {out_matrix, SIZE_X*SIZE_Y};

        #ifdef KERNEL_TIME
            // for each buf in buffers create a dummy kernel
            std::vector<buffer<float,1>> buffers;
            buffers.push_back(in_matrix_buff);
            buffers.push_back(out_matrix_buff);
            for(buffer<float,1> buf:buffers)
                utils::forceDataTransfer(Q, buf);
        #endif
        e = Q.submit([&](handler &cgh){
            // input and output amtrix accessor
            const accessor in_matrix_acc {in_matrix_buff, cgh, read_only};
            accessor out_matrix_acc {out_matrix_buff, cgh, read_write};
            // local memory with TILE_DIM+1 to avoid bank conflicts
            local_accessor<float,2> tile {range<2>{TILE_DIM, TILE_DIM+1}, cgh};
            
            cgh.parallel_for(nd_range<2>{grid, block}, matrix_transpose<SIZE_X, SIZE_Y, TILE_DIM>(
                in_matrix_acc,
                out_matrix_acc,
                tile
            )
            );//end parallel for
        });

        time_ms(e,"matrix_transpose_sycl");

    }

    #ifdef DEBUG
    for(int i = 0; i < SIZE_X*SIZE_Y; i++)
        std::cout << out_matrix[i] << ", ";
    #endif

    
    
    return 0;
}