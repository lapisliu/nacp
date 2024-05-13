cmake_minimum_required(VERSION 3.18)
project(nacp)

enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(CUBLAS REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${CUBLAS_INCLUDE_DIRS}
)

set(SOURCE_FILES
    runner.cu
    kernels/kernel1.cuh
    kernels/kernel2.cuh
    kernels/kernel3.cuh
    util.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    cuda
    cublas
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -arch=sm_70
>)
