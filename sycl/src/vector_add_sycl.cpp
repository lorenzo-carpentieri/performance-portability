

/* This example is a very small one designed to show how compact SYCL code
 * can be. That said, it includes no error checking and is rather terse. */
#include <sycl_defines.hpp>
#include <array>
#include <iostream>
#include <utils.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


namespace po = boost::program_options;


constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC, std::string use_sycl) {
  auto dev_type = utils::select_dev(use_sycl);
  auto platforms = sycl::platform::get_platforms();
  
  //Take all cpu or gpu platforms
  auto gpu_platforms = [&platforms, &dev_type](){
  std::vector<sycl::platform> gpu_platforms;
  for(auto& p : platforms){
    if(p.has(dev_type))
      if(p.get_info<sycl::info::platform::name>()==PLATFORM)
        gpu_platforms.push_back(p);
  }
      return gpu_platforms;
  }();
  
    auto device = gpu_platforms[0].get_devices()[0];

  queue deviceQueue{device};
  range<1> numOfItems{N};
  buffer<T, 1> bufferA(VA.data(), numOfItems);
  buffer<T, 1> bufferB(VB.data(), numOfItems);
  buffer<T, 1> bufferC(VC.data(), numOfItems);
  deviceQueue.submit([&](handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);
    auto kern = [=](id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    };
    cgh.parallel_for<class SimpleVadd<T>>(numOfItems, kern);
  });
}

int main(int argc, char* argv[]) {
  const size_t array_size = 4;
  std::array<int, array_size> A = {{1, 2, 3, 4}},
                                           B = {{1, 2, 3, 4}}, C;
  std::array<float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                             E = {{1.f, 2.f, 3.f, 4.f}}, F;

  std::string use_sycl="";
  po::options_description desc("Allowed options");
  desc.add_options()
      ("sycl",po::value(&use_sycl), "use SYCL implementation with cpu or gpu")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm); 
  simple_vadd(A, B, C, use_sycl);
  simple_vadd(D, E, F, use_sycl);


  for (unsigned int i = 0; i < array_size; i++) {
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
    if (F[i] != D[i] + E[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << F[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}