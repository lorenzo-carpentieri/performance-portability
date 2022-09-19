
#include <sycl/sycl.hpp>
#include <stdio.h>
#include <iostream>
#include <time_ms.hpp>
#include <sycl_defines.hpp>


#ifndef SIZE_REDUCTION
    #define SIZE_REDUCTION 30720
#endif
#define T float
int main(){
    // input data
    T *h_input = (T *) malloc(SIZE_REDUCTION * sizeof(T));
    // output
    T sumResult = 0;
    for(int i=0; i < SIZE_REDUCTION; i++)
        h_input[i]= 1;

    queue q{gpu_selector(), property::queue::enable_profiling()};
    
    event e;
    {
        // Buffers init
       
        buffer<T, 1> inBuff {h_input, SIZE_REDUCTION};
        buffer<T, 1> sumBuf { &sumResult, 1 };
    

        e = q.submit([&](handler& cgh) {

        // Input values to reductions are standard accessors
        auto inputValues = inBuff.get_access<access_mode::read>(cgh);
        auto sumValue = sumBuf.get_access<access_mode::read_write>(cgh);

        // Create temporary objects describing variables with reduction semantics
        auto sumReduction = reduction(sumValue,sycl::plus<T>());

        // parallel_for performs two reduction operations
        // For each reduction variable, the implementation:
        // - Creates a corresponding reducer
        // - Passes a reference to the reducer to the lambda as a parameter
        cgh.parallel_for(range<1>{SIZE_REDUCTION},
            sumReduction,
            [=](id<1> idx, auto& sum) {
            // plus<>() corresponds to += operator, so sum can be updated via += or combine()
            sum += inputValues[idx];
        });
        });

        time_ms(e, "reducer_sycl");

    }  
    #ifdef DEBUG
        if(sumResult==SIZE_REDUCTION){
            std::cout << "pass" << std::endl;
            return -1;
        }
        else
            std::cout <<  "fail" << std::endl;
    #endif
    return 0;

}