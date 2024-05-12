cmake_minimum_required(VERSION 3.20)

project(NACP)
enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
link_libraries(-lcurand)
FILE(
  GLOB executables
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cuh
  )

FOREACH(exe ${executables})
  GET_FILENAME_COMPONENT(filename ${exe} NAME_WE)
  add_executable(${filename} ${exe})
  set_target_properties(${filename} PROPERTIES CUDA_ARCHITECTURES "70")
  target_compile_options(
    ${filename} PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
    )
  target_include_directories(
    ${filename} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
ENDFOREACH()