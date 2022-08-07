#include <sycl/sycl.hpp>

void inline time_ms(sycl::event e, std::string kernel_name ){
            uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << kernel_name << ", " << static_cast<float>(end - start)/(float)1000000<<std::endl; 
}
