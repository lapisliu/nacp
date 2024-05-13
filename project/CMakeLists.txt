cmake_minimum_required(VERSION 3.18)
project(nacp)

enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS CUDA cuBLAS cuBLASLt)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${CUBLAS_INCLUDE_DIRS}
)

set(SOURCE_FILES
    runner.cu
    kernels/wmma-mix.cuh
    kernels/cublas-mix.cuh
    kernels/basic_tiling.cuh
    util.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
set_target_properties(nacp PROPERTIES
    CUDA_ARCHITECTURES 70
)
target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    cuda
    cublas
    cublasLt
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
