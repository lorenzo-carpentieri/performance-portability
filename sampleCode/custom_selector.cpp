#include <sycl/sycl.hpp>
#include <stdio.h>
#include <iostream>

using namespace sycl;

class special_device_selector : public device_selector
{
public:
  int operator()(const sycl::device& dev) const override
  {
    if (dev.is_gpu()) {
      auto vendorName = dev.get_info<sycl::info::device::vendor>();
      if (vendorName.find("NVIDIA") != std::string::npos) {
        return 1;
      }
    }
    return -1;
  }
};


int main(){
    device d;
    queue q {special_device_selector()};
    // Take context, platform and device
    context ctx = q.get_context();
    platform pt = ctx.get_platform();
    device dev = q.get_device();
    
    std::cout<< dev.get_info<info::device::name>() << std::endl;
  

}