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

    std::vector<std::vector<int>> out_vecs;


    in_vecA.resize(N);
    in_vecB.resize(N);
    out_vecs.resize(4);

    for(int i = 0; i < 4; i++)
        out_vecs[i].resize(N);


    std::fill(in_vecA.begin(), in_vecA.end(), 1);
    std::fill(in_vecB.begin(), in_vecB.end(), 2);

    for(int i = 0; i < 4; i++)
        std::fill(out_vecs[i].begin(), out_vecs[i].end(), 0);

    queue Q;
    {
        event e1, e2, e3, e4;
        range<1> nd_range{N};
        // Device allocation
        int *in_A = malloc_device<int>(N, Q);
        int *in_B = malloc_device<int>(N, Q);
        
        int *out_C = malloc_device<int>(N, Q);
        int *out_D = malloc_device<int>(N, Q);
        int *out_E = malloc_device<int>(N, Q);
        int *out_F = malloc_device<int>(N, Q);

        Q.memcpy(in_A, &in_vecA[0], (N)*sizeof(int)); 
        Q.memcpy(in_B, &in_vecB[0], (N)*sizeof(int));  

        Q.memcpy(out_C, &out_vecs[0], (N)*sizeof(int)); 
        Q.memcpy(out_D, &out_vecs[1], (N)*sizeof(int));
        Q.memcpy(out_E, &out_vecs[2], (N)*sizeof(int)); 
        Q.memcpy(out_F, &out_vecs[3], (N)*sizeof(int));  
       

        e1 = Q.parallel_for(nd_range, VecAdd(
                in_A,
                in_B,
                out_C
            ));// end parallel_for

        e2 = Q.parallel_for(nd_range,{e1}, VecAdd(
                out_C,
                in_A,
                out_D
            ));// end parallel_for

        e3 = Q.parallel_for(nd_range,{e1}, VecAdd(
                out_C,
                in_B,
                out_E
            ));// end parallel_for

        e4 = Q.parallel_for(nd_range, {e2, e3},VecAdd(
                out_D,
                out_E,
                out_F
            ));// end parallel_for
        Q.wait();
        Q.memcpy(&out_vecs[3][0], out_F, (N)*sizeof(int));
        Q.wait();
        
        sycl::free(in_A,Q);
        sycl::free(in_B,Q);
        sycl::free(out_C,Q);
        sycl::free(out_D,Q);
        sycl::free(out_E,Q);
        sycl::free(out_F,Q);

    } //end block
    
    for(int i = 0; i < N; i++)
        if(out_vecs[3][i] != 9)
            std::cout<< "Fail" << std::endl;
    
    std::cout<< "Pass"<< std::endl;    
}