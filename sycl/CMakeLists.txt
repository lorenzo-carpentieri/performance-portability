
cmake_minimum_required(VERSION 3.22)
project(Tesi_sycl)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/sycl)
# list exe
set(EXECUTABLES nbody vector_add matrix_transpose matrix_mul_tiling class_reducer reduction_subgroup reduction_binary  box_blur box_blur_local_memory sobel_filter)
#class_reducer    (Controllare se è supportata da dpcpp)
# set(EXECUTABLES  vector_add  matrix_mul_tiling class_reducer reduction_subgroup reduction_binary box_blur box_blur_local_memory sobel_filter)
# controllare nbody
foreach(EXE IN LISTS EXECUTABLES)
	add_executable(${EXE} ${CMAKE_CURRENT_SOURCE_DIR}/include/utils.hpp ${CMAKE_CURRENT_SOURCE_DIR}/include/time_ms.hpp ${CMAKE_CURRENT_SOURCE_DIR}/include/sycl_defines.hpp ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)


	target_include_directories(${EXE}
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
	)
	target_sources(${EXE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/${EXE}_sycl.cpp
	# ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_app.cpp
	)

	# add_compile_definitions(${EXE}
	# CPU_DEVICE=${CPU_DEVICE}
	# GPU_DEVICE=${GPU_DEVICE}
	# HOST_DEVICE=${HOST_DEVICE}
	# DEVICE_VALUE=${device_value}
	# )
		
	if(SYCL_BACKEND STREQUAL "hipSYCL")
		find_package(hipSYCL REQUIRED)
		add_definitions(-DHIPSYCL)
		add_sycl_to_target(TARGET ${EXE})
	endif()

  	if(SYCL_BACKEND STREQUAL "dpcpp")
  		add_definitions (-DDPCPP)
  	endif()
	
	if(KERNEL_PROFILING)
	  add_definitions(-DKERNEL_TIME)
  	endif()

	if(DEBUG)
	  add_definitions(-DDEBUG)
  	endif()
	
	add_definitions(-DPLATFORM="${PLATFORM}")
endforeach()


# Set compile time variable
# target_compile_definitions(nbody PRIVATE SIZE_BODY=15104)
add_compile_definitions(
	MATRIX_SIZE=2048 
	SIZE_BODY=20480
	SIZE_X=8198 SIZE_Y=8192 #matrix_transpose sizes
	SIZE_REDUCTION=245760
	RADIUS=3
    RADIUS_SOBEL=3
						)
option(SYCL_BACKEND "Enable selected SYCL backend" "")
option(SYCL_DEVICE "Select target SYCL device" GPU)

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



message(STATUS "---- SYCL CONFIG ----")
message(STATUS "SYCL_BACKEND: ${SYCL_BACKEND}")
message(STATUS "SYCL_DEVICE: ${SYCL_DEVICE}")
message(STATUS "---- SYCL CONFIG ----\n")
