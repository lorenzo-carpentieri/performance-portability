#include <sycl/sycl.hpp>
#ifndef PLATFORM
    #error Platform not defined
#endif

namespace utils{   
    sycl::aspect inline select_dev(std::string type){
         if (type == "cpu") {
          return sycl::aspect::cpu;
        } else if (type == "gpu") {
          return sycl::aspect::gpu;
        } else {
          throw std::runtime_error("Unknown device type: " + type);
        }
    }

    // Dummy kernel to force data transfer
    template<class AccType>
    class InitializationDummyKernel
    {
    public:
      InitializationDummyKernel(AccType acc)
      : acc{acc} {}
    
      void operator()() const {}
    private:
      AccType acc;
    };

    template <class BufferType>
    inline void forceDataTransfer(sycl::queue& q, BufferType b) {
      q.submit([&](sycl::handler& cgh) {
        auto acc = b.template get_access<sycl::access::mode::read>(cgh);
        cgh.single_task(InitializationDummyKernel{acc});
      });
      q.wait_and_throw();
    }

    
}

  