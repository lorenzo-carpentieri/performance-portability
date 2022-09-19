
#include <sycl/sycl.hpp>

using namespace sycl;
#ifdef DPCPP
    
    template <typename T, int dimensions>
    using local_accessor =
        accessor<T, dimensions, access::mode::read_write, access::target::local>;
#endif 
    


