#include <sycl/sycl.hpp>
#include <stdio.h>
#include <iostream>

using namespace sycl;
int main(){
    
    queue q {gpu_selector()};
    // Take context, platform and device
    context ctx = q.get_context();
    platform pt = ctx.get_platform();
    device dev = q.get_device();
    
    std::vector<device> devices = ctx.get_info<info::context::devices>();
    std::vector<platform> platforms = pt.get_platforms();
    // Plaforms info
    std::cout<< "Platforms info" << std::endl;

    for(int i = 0; i < platforms.size(); i++)
        std::cout << "Platform name: " <<  platforms[i].get_info<info::platform::name>()<<std::endl;

    std::cout<<"\nDevices Info"<< std::endl;
    // print devices info
    for(int i = 0; i < devices.size(); i++){
        std::cout << "Device name: " << devices[i].get_info<info::device::name>() << std::endl;
        std::cout << "Device vendor name: " << devices[i].get_info<info::device::vendor>() << std::endl;
        std::cout << "Device platform name: " << devices[i].get_info<info::device::platform>().get_info<info::platform::name>() << std::endl;

        std::cout << "Max compute units: " << devices[i].get_info<info::device::max_compute_units>() << std::endl; 
        std::cout << "Max Work Item Dimension (1-3): " << devices[i].get_info<info::device::max_work_item_dimensions>() << std::endl; 
       
        id<3> id = devices[i].get_info<info::device::max_work_item_sizes>();
        std::cout << "Max Work item sizes: " << id.get(0) << ", "<< id.get(1) << ", "<< id.get(2) << std::endl; 
        std::cout << "Max Work group size: " << devices[i].get_info<info::device::max_work_group_size>() << std::endl; 
        std::cout << "Max num sub group: " << devices[i].get_info<info::device::max_num_sub_groups>() << std::endl; 
        
        std::vector<size_t> sub_group_sizes = devices[i].get_info<info::device::sub_group_sizes>();
        for(int j = 0; j < sub_group_sizes.size(); j++)
            std::cout << "Max sub group size: " << sub_group_sizes[j] << std::endl; 


        std::cout << "" << std::endl;

        
    }


}