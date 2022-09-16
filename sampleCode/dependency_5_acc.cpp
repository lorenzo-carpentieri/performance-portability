#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;
class VecAdd{
  private:
    const accessor<int, 1, sycl::access_mode::read, sycl::target::device> a;
    const accessor<int, 1, sycl::access_mode::read, sycl::target::device> b;
    accessor<int, 1, sycl::access_mode::read_write, sycl::target::device> c;

  public:
    VecAdd( 
      const accessor<int, 1, sycl::access_mode::read, sycl::target::device> a,
      const accessor<int, 1, sycl::access_mode::read, sycl::target::device> b,
      accessor<int, 1, sycl::access_mode::read_write, sycl::target::device> c)
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
        auto start_init = std::chrono::steady_clock::now();
        auto start = std::chrono::steady_clock::now();


        // input vec 
        buffer<int, 1> in_bufA{in_vecA.data(), range<1>{N}};
        buffer<int, 1> in_bufB{in_vecB.data(), range<1>{N}};
        //output vec
        buffer<int, 1> out_buf0{out_vecs[0].data(), range<1>{N}};
        buffer<int, 1> out_buf1{out_vecs[1].data(), range<1>{N}};
        buffer<int, 1> out_buf2{out_vecs[2].data(), range<1>{N}};
        buffer<int, 1> out_buf3{out_vecs[3].data(), range<1>{N}};


       
        e1 = Q.submit([&](handler &cgh){
            accessor<int, 1> in_A {in_bufA, cgh, read_only};
            accessor<int, 1> in_B {in_bufB, cgh, read_only};
            accessor<int, 1> out {out_buf0, cgh, read_write, no_init};

            range<1> nd_range{N};
            cgh.parallel_for(nd_range, VecAdd(
                in_A,
                in_B,
                out
            ));// end parallel_for
        });// end submits

        e2 = Q.submit([&](handler &cgh){
            accessor<int, 1> in_A {out_buf0, cgh, read_only};
            accessor<int, 1> in_B {in_bufA, cgh, read_only};
            accessor<int, 1> out {out_buf1, cgh, read_write, no_init};

            range<1> nd_range{N};
            cgh.parallel_for(nd_range, VecAdd(
                in_A,
                in_B,
                out
            ));// end parallel_for
        });// end submits

        e3 = Q.submit([&](handler &cgh){
            accessor<int, 1> in_A {out_buf0, cgh, read_only};
            accessor<int, 1> in_B {in_bufB, cgh, read_only};
            accessor<int, 1> out {out_buf2, cgh, read_write, no_init};

            range<1> nd_range{N};
            cgh.parallel_for(nd_range, VecAdd(
                in_A,
                in_B,
                out
            ));// end parallel_for
        });// end submits

        e4 = Q.submit([&](handler &cgh){
            accessor<int, 1> in_A {out_buf1, cgh, read_only};
            accessor<int, 1> in_B {out_buf2, cgh, read_only};
            accessor<int, 1> out {out_buf3, cgh, read_write, no_init};

            range<1> nd_range{N};
            cgh.parallel_for(nd_range, VecAdd(
                in_A,
                in_B,
                out
            ));// end parallel_for
        });// end submits
    } //end block
    
    

    for(int i = 0; i < N; i++)
        if(out_vecs[3][i] != 9)
            std::cout<< "Fail" << std::endl;
    
    std::cout<< "Pass"<< std::endl;    
}