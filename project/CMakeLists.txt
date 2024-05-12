cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MyCUDAProject LANGUAGES CXX CUDA)

# Set a default CUDA architecture if not specified
set(DEFAULT_CUDA_ARCH "sm_70")
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${DEFAULT_CUDA_ARCH} CACHE STRING "CUDA architecture" FORCE)
endif()

# Include directories
include_directories(include)

# Find CUDA
find_package(CUDA REQUIRED)

# Add .cu and .cuh files
file(GLOB_RECURSE CUDA_SOURCES "kernels/*.cu")
file(GLOB_RECURSE HEADERS "include/*.h")

# Executable
add_executable(${PROJECT_NAME} runner.cu ${CUDA_SOURCES} ${HEADERS})

# Link libraries (if any)
# target_link_libraries(${PROJECT_NAME} <library_name>)

# Specify C++ standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
