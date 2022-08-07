
cmake_minimum_required(VERSION 3.22)
project(Tesi_sycl)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/sycl)
# list exe
set(EXECUTABLES nbody vector_add matrix_transpose matrix_mul_tiling reduction_subgroup)

foreach(EXE IN LISTS EXECUTABLES)
	add_executable(${EXE})
	target_include_directories(${EXE}
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
	)
	target_sources(${EXE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/${EXE}_sycl.cpp
	# ${CMAKE_CURRENT_SOURCE_DIR}/src/sample_app.cpp
	)

	add_compile_definitions(${EXE}
	CPU_DEVICE=${CPU_DEVICE}
	GPU_DEVICE=${GPU_DEVICE}
	HOST_DEVICE=${HOST_DEVICE}
	DEVICE_VALUE=${device_value}
	)

	add_compile_definitions(${EXE}
		CPU_DEVICE=${CPU_DEVICE}
		GPU_DEVICE=${GPU_DEVICE}
		HOST_DEVICE=${HOST_DEVICE}
		DEVICE_VALUE=${device_value}
	)
		
	if(SYCL_BACKEND STREQUAL hipSYCL)
		find_package(hipSYCL REQUIRED)
		add_sycl_to_target(TARGET ${EXE})
		add_sycl_to_target(TARGET ${EXE})
	endif()

endforeach()

add_executable(box_blur ${CMAKE_CURRENT_SOURCE_DIR}/src/box_blur_sycl.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)
add_executable(box_blur_local_memory ${CMAKE_CURRENT_SOURCE_DIR}/src/box_blur_local_memory_sycl.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp)
add_executable(sobel_filter ${CMAKE_CURRENT_SOURCE_DIR}/src/sobel_filter_sycl.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/lodepng.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/time_ms.hpp)



if(SYCL_BACKEND STREQUAL hipSYCL)
		find_package(hipSYCL REQUIRED)
		add_sycl_to_target(TARGET box_blur)
endif()
if(SYCL_BACKEND STREQUAL hipSYCL)
		find_package(hipSYCL REQUIRED)
		add_sycl_to_target(TARGET box_blur_local_memory)
endif()
if(SYCL_BACKEND STREQUAL hipSYCL)
		find_package(hipSYCL REQUIRED)
		add_sycl_to_target(TARGET sobel_filter)
endif()

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
