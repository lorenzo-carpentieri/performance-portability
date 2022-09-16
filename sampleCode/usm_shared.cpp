#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

class VecAdd{
  private:
    const int *a;
    const int *b;
    int *c;

  public:
    VecAdd(
          const int *a,
          const int *b,
          int *c)
      :
      a(a),
      b(b),
      c(c)
    {}
    void operator()(id<1> id)const{
      c[id] = a[id] + b[id];
    }
};


int main (int argc, char **argv){
    
    size_t N = 1000;
    
    std::vector<int> in_vecA;
    std::vector<int> in_vecB;

    std::vector<int> out_vec;

    in_vecA.resize(N);
    in_vecB.resize(N);
    out_vec.resize(N);

    std::fill(in_vecA.begin(), in_vecA.end(), 1);
    std::fill(in_vecB.begin(), in_vecB.end(), 1);
    std::fill(out_vec.begin(), out_vec.end(), 0);


    queue Q{};
    
    event e;

    // device allocation
    int *in_A = malloc_shared<int>(N, Q);
    int *in_B = malloc_shared<int>(N, Q);
    int *out = malloc_shared<int>(N, Q);

   for(int i = 0; i < N; i++){
        in_A[i] = 1;
        in_B[i] = 1;
        out[i] = 0;
   }
    
    Q.wait();
    
   
        // Computation on device
        e = Q.parallel_for(range<1>{N},VecAdd(
            in_A,
            in_B,
            out
        ));

        Q.wait();
        
    // print result
     for(int i = 0; i < N; i++){
        if(out[i]!=2)
            std::cout << "Fail" << std::endl;
    }
    
    std::cout << "Pass" << std::endl;
    
    sycl::free(in_A,Q);
    sycl::free(in_B,Q);
    sycl::free(out,Q);
 
    
}