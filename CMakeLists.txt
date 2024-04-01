cmake_minimum_required(VERSION 3.10)

project(fluxxx)

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_COMPILER mpic++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_PREFIX_PATH "libtorch")

find_package(Torch REQUIRED)
find_package(PythonLibs 3.8 REQUIRED)
find_package(MPI REQUIRED)

file(GLOB SOURCES "src/*.cpp")

include_directories(include)
include_directories(/home/konstantin/Desktop/master/fluxxx/python/include/python3.8)
include_directories(/home/konstantin/Desktop/master/fluxxx/python/lib/python3.8/site-packages/numpy/core/include)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(fluxxx ${SOURCES})

target_precompile_headers(fluxxx PRIVATE <torch/torch.h>)
target_precompile_headers(fluxxx PRIVATE <Python.h>)

target_link_libraries(fluxxx "${TORCH_LIBRARIES}")
target_link_libraries(fluxxx "/home/konstantin/Desktop/master/fluxxx/python/lib/libpython3.8.so")
