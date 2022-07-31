cmake_minimum_required(VERSION 3.22)
project(Tesi_cuda)

include(CheckLanguage)
check_language(CUDA)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda)
enable_language(CUDA)

# set(CMAKE_CUDA_FLAGS "-lcublas")
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

# target_compile_options(matrix_mul
#     PUBLIC
#     $<$<COMPILE_LANGUAGE:CUDA>: -lcublas>
# )


message(STATUS "---- CUDA CONFIG ----")
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "---- CUDA CONFIG ----\n")
