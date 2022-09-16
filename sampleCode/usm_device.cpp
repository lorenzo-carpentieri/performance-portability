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
    {
    event e;

    // device allocation al posto dei buffer
    int *in_A = malloc_device<int>(N, Q);
    int *in_B = malloc_device<int>(N, Q);
    int *out = malloc_device<int>(N, Q);

    //init device 
    Q.memcpy(in_A, &in_vecA[0], (N)*sizeof(int)); 
    Q.memcpy(in_B, &in_vecB[0], (N)*sizeof(int));   
    Q.memcpy(out, &out_vec[0], (N)*sizeof(int)); 
    
    Q.wait();
    
        // Computation on device
        e = Q.parallel_for(range<1>{N},VecAdd(
            in_A,
            in_B,
            out
        ));

        Q.wait();
            
        Q.memcpy(&out_vec[0], out, (N)*sizeof(int));

        Q.wait();      
        sycl::free(in_A,Q);
        sycl::free(in_B,Q);
        sycl::free(out,Q);
    }//end block

 
    // print result
     for(int i = 0; i < N; i++){
        std::cout << out_vec[i] << std::endl;
    }
    
}