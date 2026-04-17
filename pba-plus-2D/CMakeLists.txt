cmake_minimum_required(VERSION 3.20)

project(PBA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(pba "main.cpp" "pba/pba2D.h" "pba/pba2DHost.cu" "pba/pba2DKernel.h")

# Link necessary libraries
target_link_libraries(pba PRIVATE
        CUDA::cudart # CUDA Runtime library
        CUDA::nppc # NPP Core library
        CUDA::nppif  # NPP image filtering
        CUDA::nppidei # NPP Data Exchange and Initialization library
        CUDA::nppisu # NPP Image Support library
)
