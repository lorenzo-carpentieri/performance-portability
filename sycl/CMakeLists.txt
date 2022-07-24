
cmake_minimum_required(VERSION 3.22)
project(Tesi_sycl)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/sycl)

#executables list
add_executable(vector_add)
add_executable(nbody)

option(SYCL_BACKEND "Enable selected SYCL backend" "")
option(SYCL_DEVICE "Select target SYCL device" GPU)

target_include_directories(vector_add
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(vector_add PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/vector_add.cpp
	# ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_app.cpp
)
target_sources(nbody PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/nbody_sycl.cpp
)
# set_target_properties(vector_add PROPERTIES
# 	PUBLIC_HEADER
# 	${CMAKE_CURRENT_SOURCE_DIR}/include/vector_add.hpp
# )

# Handle multiple devices
set(CPU_DEVICE 0)
set(GPU_DEVICE 1)
set(HOST_DEVICE 2)

string(TOUPPER ${SYCL_DEVICE} SYCL_DEVICE)
set(device_value -1)
if(SYCL_DEVICE STREQUAL CPU)
	set(device_value ${CPU_DEVICE})
elseif(SYCL_DEVICE STREQUAL GPU)
	set(device_value ${GPU_DEVICE})
elseif(SYCL_DEVICE STREQUAL HOST)
	set(device_value ${HOST_DEVICE})
else()
	message(FATAL_ERROR "Unknown device selected: ${SYCL_DEVICE}")
endif()

add_compile_definitions(vector_add
	CPU_DEVICE=${CPU_DEVICE}
	GPU_DEVICE=${GPU_DEVICE}
	HOST_DEVICE=${HOST_DEVICE}
	DEVICE_VALUE=${device_value}
)

add_compile_definitions(nbody
	CPU_DEVICE=${CPU_DEVICE}
	GPU_DEVICE=${GPU_DEVICE}
	HOST_DEVICE=${HOST_DEVICE}
	DEVICE_VALUE=${device_value}
)


if(SYCL_BACKEND STREQUAL hipSYCL)
	find_package(hipSYCL REQUIRED)
	add_sycl_to_target(TARGET vector_add)
	add_sycl_to_target(TARGET nbody)
endif()


message(STATUS "---- SYCL CONFIG ----")
message(STATUS "SYCL_BACKEND: ${SYCL_BACKEND}")
message(STATUS "SYCL_DEVICE: ${SYCL_DEVICE}")
message(STATUS "---- SYCL CONFIG ----\n")
