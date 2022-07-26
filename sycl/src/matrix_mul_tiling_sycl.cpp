#include <stdio.h>
#include <iostream>
#include <sycl_defines.hpp>
#include "time_ms.hpp"

#include <utils.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


namespace po = boost::program_options;

#ifndef MATRIX_SIZE
    #define MATRIX_SIZE 1024
#endif
#define BLOCK_SIZE 32



//only square matrix with MATRIX_SIZE multple of BLOCK_SIZE
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

            int local_id_x = it.get_local_id(1);
            int local_id_y = it.get_local_id(0);
            
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


int main(int argc, char* argv[])
{
    
    // Computation is divided into tiles of TILE_DIM X TILE_DIME (where TILE_DIM is multiple of BLOCK_ROWS). 
    // execution configuration parameters
 
   
    
    // size of memory required to store the matrix
    const int mem_size = sizeof(float) * MATRIX_SIZE*MATRIX_SIZE;
    
   
    float *a_matrix = (float*) malloc(mem_size);
    float *b_matrix = (float*) malloc(mem_size);
    float *c_matrix = (float*) malloc(mem_size);

  
    for(int i = 0; i < (MATRIX_SIZE*MATRIX_SIZE); ++i){
        a_matrix[i] = (float) 1;
        b_matrix[i] = (float) 1;
        c_matrix[i] = 0.0;
    }

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
    std::string platform_name;
    if(p.has(dev_type))
      platform_name = p.get_info<sycl::info::platform::name>();
      if(platform_name.find(PLATFORM)!=-1)
        gpu_platforms.push_back(p);
  }
      return gpu_platforms;
  }();
  
    auto device = gpu_platforms[0].get_devices()[0];


    queue Q{device, property::queue::enable_profiling()};

    {
        event e;
        range<2> grid {MATRIX_SIZE, MATRIX_SIZE}; 
        range<2> block{BLOCK_SIZE, BLOCK_SIZE};
        buffer<float, 1> a_matrix_buff {a_matrix, MATRIX_SIZE*MATRIX_SIZE};
        buffer<float, 1> b_matrix_buff {b_matrix, MATRIX_SIZE*MATRIX_SIZE};
        buffer<float, 1> c_matrix_buff {c_matrix, MATRIX_SIZE*MATRIX_SIZE};
        
        #ifdef KERNEL_TIME
            // for each buf in buffers create a dummy kernel
            std::vector<buffer<float,1>> buffers;
            buffers.push_back(a_matrix_buff);
            buffers.push_back(b_matrix_buff);
            buffers.push_back(c_matrix_buff);
            for(buffer<float,1> buf:buffers)
                utils::forceDataTransfer(Q, buf);
        #endif

        e = Q.submit([&](handler &cgh){
            // input and output amtrix accessor
            const accessor a_matrix_acc {a_matrix_buff, cgh, read_only};
            const accessor b_matrix_acc {b_matrix_buff, cgh, read_only};
            accessor c_matrix_acc {c_matrix_buff, cgh, read_write};

            // local accessor for tile
            local_accessor<float, 2> tile_a{range<2>{BLOCK_SIZE, BLOCK_SIZE}, cgh};
            local_accessor<float, 2> tile_b{range<2>{BLOCK_SIZE, BLOCK_SIZE}, cgh};

            cgh.parallel_for(nd_range<2>{grid, block}, square_matrix_mul_tiling<MATRIX_SIZE>(
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
    for(int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++)
        if(c_matrix[i]!= MATRIX_SIZE)
            std::cout << "fail" << std::endl;
    std::cout<< "pass" <<  std::endl;
    #endif
    return 0;
}