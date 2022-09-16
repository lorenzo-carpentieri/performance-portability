cmake_minimum_required(VERSION 3.22)
project(Tesi_cuda)

include(CheckLanguage)
check_language(CUDA)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

enable_language(CUDA)

set(EXECUTABLES nbody matrix_mul_cublas matrix_mul_tiling matrix_transpose reduction_tile32 reduction_warp_primitive)

foreach(EXE IN LISTS EXECUTABLES)
	add_executable(${EXE})
	target_include_directories(${EXE}
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
	)
	target_sources(${EXE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/${EXE}_cuda.cu
	# ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_app.cpp
	)
endforeach()
# Generate ptx 
# add_library(prova OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/prova.cu  ${CMAKE_CURRENT_SOURCE_DIR}/src/prova.cuh)
# set_property(TARGET prova PROPERTY CUDA_PTX_COMPILATION ON)
# add_library(reduction_tile32 OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/reduction_tile32_cuda.cu)
# set_property(TARGET reduction_tile32 PROPERTY CUDA_PTX_COMPILATION ON)

# Generate ptx
# target_compile_options(
#     reduction_warp_primitive
#     PRIVATE 
#     "--keep"
# )

add_executable(box_blur_local_memory ${CMAKE_CURRENT_SOURCE_DIR}/src/box_blur_local_memory_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)
add_executable(box_blur ${CMAKE_CURRENT_SOURCE_DIR}/src/box_blur_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)
add_executable(sobel_filter ${CMAKE_CURRENT_SOURCE_DIR}/src/sobel_filter_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)
add_executable(box_blur_bidim ${CMAKE_CURRENT_SOURCE_DIR}/src/box_blur_bidim_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)






message(STATUS "---- CUDA CONFIG ----")
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "---- CUDA CONFIG ----\n")
