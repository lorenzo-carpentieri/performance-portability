
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "time_ms.hpp"
#include <sycl_defines.hpp>

#include <utils.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


namespace po = boost::program_options;

#ifndef SIZE_BODY
    #define SIZE_BODY 30208
#endif
#define BLOCK_SIZE 256
#define DELTA_TIME 0.2f
#define EPS2  1e-9f
#define NUM_TILES (SIZE_BODY + BLOCK_SIZE - 1) / BLOCK_SIZE



sycl::float3 bodyBodyInteraction(sycl::float4 bi, sycl::float4 bj, sycl::float3 ai)
{
    sycl::float4 r;
    // r_ij [3 FLOPS]
    r = bj - bi;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + EPS2;

    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
    
    // s = m_j * invDistCube [1 FLOP]
    float s = bj[3] * invDistCube;
    
    // a_i = a_i + s * r_ij [6 FLOPS]
    sycl::float3 tmp  {r[0]*s, r[1]*s, r[2]*s};
    ai =  ai + tmp;
    
    return ai;
}

template<int TILE_SIZE>
sycl::float3 tile_calculation(sycl::float4 myPosition, sycl::float3 accel, local_accessor<sycl::float4, 1> sh_position)
{

    for (int i = 0; i < TILE_SIZE; i++) {
        accel = bodyBodyInteraction(myPosition, sh_position[i], accel);
    }
    return accel;
}



template<int TILE_SIZE, int TILES>
class calculate_forces{
    private:
        const accessor<sycl::float4, 1, access_mode::read, target::device> in_pos;
        const accessor<sycl::float4, 1, access_mode::read, target::device> in_vel;
        accessor<sycl::float4, 1, access_mode::read_write, target::device> out_pos;
        accessor<sycl::float4, 1, access_mode::read_write, target::device> out_vel;
        local_accessor<sycl::float4, 1> sh_position;
    
    public:
        calculate_forces(
            const accessor<sycl::float4, 1, access_mode::read, target::device> in_pos,
            const accessor<sycl::float4, 1,access_mode::read, target::device> in_vel,
            accessor<sycl::float4, 1, access_mode::read_write, target::device> out_pos,
            accessor<sycl::float4, 1, access_mode::read_write, target::device> out_vel,
            local_accessor<sycl::float4, 1> sh_position
        )
        :
            in_pos(in_pos),
            in_vel(in_vel),
            out_pos(out_pos),
            out_vel(out_vel),
            sh_position(sh_position){}

        void operator()(sycl::nd_item<1> it) const{
            const auto &group = it.get_group();
            int gtid = it.get_global_id().get(0);
            int local_id = it.get_local_id().get(0);

            sycl::float4 myPosition;
            sycl::float3 acc = {0.0f, 0.0f, 0.0f};
        
            myPosition = in_pos[gtid];

    

            for (int i = 0, tile = 0; i < TILES; i++, tile++) {
                int idx = tile * BLOCK_SIZE + local_id;
                sh_position[local_id] = in_pos[idx];
                sycl::group_barrier(group);
            
                acc = tile_calculation<TILE_SIZE>(myPosition, acc, sh_position);

                sycl::group_barrier(group);
            }
            // Save the result in global memory for the integration step.
            sycl::float4 acc4 = {acc[0], acc[1], acc[2], 0.0f};

            sycl::float4 oldVel;
            oldVel = in_vel[gtid];
            // updated position and velocity
            sycl::float4 newPos = myPosition + oldVel * DELTA_TIME + acc4 * 0.5f * DELTA_TIME * DELTA_TIME;
            newPos[3] = myPosition[3];
            sycl::float4 newVel = oldVel + (acc4 * DELTA_TIME);
            // write to global memory
            out_pos[gtid] = newPos;
            out_vel[gtid] = newVel;
        }
};

int main(int argc, char* argv[]) {    
    
    int bytes = SIZE_BODY * sizeof(sycl::float4);

    sycl::float4 *pos = (sycl::float4*)malloc(bytes);
    sycl::float4 *vel = (sycl::float4*)malloc(bytes);
    
    sycl::float4 *new_pos = (sycl::float4*)malloc(bytes);
    sycl::float4 *new_vel = (sycl::float4*)malloc(bytes);
    
    // Initialization bodies pos and vel
    srand(10);
    for(int i = 0; i < SIZE_BODY; i++){
        // pos
        pos[i][0]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i][1]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i][2]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        // mass
        pos[i][3]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        //vel
        vel[i][0]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i][1]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i][2]= 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        vel[i][3]= 0;
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

    sycl::queue Q {device, sycl::property::queue::enable_profiling()};
    
    {
        // buffer
        sycl::buffer<sycl::float4,1> pos_buff {pos, SIZE_BODY};
        sycl::buffer<sycl::float4, 1> vel_buff {vel, SIZE_BODY};
        sycl::buffer<sycl::float4, 1> out_vel_buff {new_vel, SIZE_BODY};
        sycl::buffer<sycl::float4, 1> out_pos_buff{ new_pos, SIZE_BODY};
        
        #ifdef KERNEL_TIME
            // for each buf in buffers create a dummy kernel
            std::vector<buffer<sycl::float4,1>> buffers;
            buffers.push_back(pos_buff);
            buffers.push_back(vel_buff);
            buffers.push_back(out_vel_buff);
            buffers.push_back(out_pos_buff);

            for(buffer<sycl::float4,1> buf:buffers)
                utils::forceDataTransfer(Q, buf);
        #endif
        sycl::range<1> block{BLOCK_SIZE};
        sycl::range<1> grid{SIZE_BODY}; 
        sycl::event e;
    
        e = Q.submit([&](sycl::handler &cgh) {
            // accessor
            const accessor in_pos{pos_buff, cgh, sycl::read_only};
            const accessor in_vel{vel_buff, cgh, sycl::read_only};
            accessor out_pos{out_pos_buff, cgh, sycl::read_write};
            accessor out_vel{out_vel_buff, cgh, sycl::read_write};
            local_accessor<sycl::float4,1> sh_position{sycl::range<1>{BLOCK_SIZE},cgh};

            cgh.parallel_for(sycl::nd_range<1>(grid, block), calculate_forces<BLOCK_SIZE, NUM_TILES>(
                in_pos,
                in_vel,
                out_pos,
                out_vel,
                sh_position
            ));
        });
        time_ms(e, "nbody_sycl");
    }

    #ifdef DEBUG
    // print results
    for(int i = 0; i < SIZE_BODY; i++){
        printf("body: %d, new_pos_x: %.2f, new_pos_y: %.2f, new_pos_z: %.2f\n",  i, new_pos[i][0], new_pos[i][1], new_pos[i][2]);
        printf("body: %d, new_vel_x: %.2f, new_vel_y: %.2f, new_vel_z: %.2f\n",  i, new_vel[i][0], new_vel[i][1], new_vel[i][2]);
        printf("\n");
    }
    #endif    
}

