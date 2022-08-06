cmake_minimum_required(VERSION 3.22)
project(Tesi_cuda)

include(CheckLanguage)
check_language(CUDA)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda)
enable_language(CUDA)

set(EXECUTABLES nbody matrix_mul_cublas matrix_mul_tiling matrix_transpose reduction_tile32)

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

add_executable(box_blur_local_memory ${CMAKE_CURRENT_SOURCE_DIR}/src/box_blur_local_memory_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)
add_executable(box_blur ${CMAKE_CURRENT_SOURCE_DIR}/src/box_blur_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)




message(STATUS "---- CUDA CONFIG ----")
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "---- CUDA CONFIG ----\n")
