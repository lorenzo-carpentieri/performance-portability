#include <sycl/sycl.hpp>
#include <array>
#include <iostream>

using namespace sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;


template <typename T>
class SimpleVadd;


template <typename T, size_t N>
void simple_vadd(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC) {
  queue deviceQueue;
  range<1> numOfItems{N};
  buffer<T, 1> bufferA(VA.data(), numOfItems);
  buffer<T, 1> bufferB(VB.data(), numOfItems);
  buffer<T, 1> bufferC(VC.data(), numOfItems);
  deviceQueue.submit([&](handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);
    
    cgh.parallel_for(
            range<1>{N}, // 1-dimensional index space with N work-items
            [=](item<1> it){
                int id = it.get_id();
                accessorC[id] = accessorA[id] + accessorB[id];
            } 
    );// end parallel for
  }); // end submit
}

int main() {
  const size_t array_size = 4;
  std::array<int, array_size> A = {{1, 2, 3, 4}},
                              B = {{1, 2, 3, 4}}, 
                              C;

  std::array<int, array_size> D = {{1, 2, 3, 4}},
                                E = {{1, 2, 3, 4}}, 
                                F;
  simple_vadd(A, B, C);
  simple_vadd(D, E, F);


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