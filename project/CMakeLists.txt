cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MyCUDAProject LANGUAGES CXX CUDA)

# Specify the CUDA architecture
set(CUDA_ARCHITECTURE "70" CACHE STRING "CUDA architecture")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE})

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Find CUDA
find_package(CUDA REQUIRED)

# Add .cuh and .h files
file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_SOURCE_DIR}/kernels/*.cuh")
file(GLOB_RECURSE HEADERS "${CMAKE_SOURCE_DIR}/include/*.h")

# Executable
add_executable(${PROJECT_NAME} runner.cu ${CUDA_SOURCES} ${HEADERS})

# Link libraries (if any)
# target_link_libraries(${PROJECT_NAME} -lcublas -lcublasLt)

# Specify C++ standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
