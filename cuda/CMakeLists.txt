cmake_minimum_required(VERSION 3.22)
project(Tesi_cuda)
include(CheckLanguage)
check_language(CUDA)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cuda)
enable_language(CUDA)

add_executable(nbody)

target_include_directories(nbody
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(nbody PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/nbody_cuda.cu
	# ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_app.cpp
)
message(STATUS "---- CUDA CONFIG ----")
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "---- CUDA CONFIG ----\n")
