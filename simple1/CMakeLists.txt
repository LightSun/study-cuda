cmake_minimum_required(VERSION 3.5)

project(cuda_simple1 LANGUAGES C CXX)

#libtorch require cpp14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -g")

set(LIB_TORCH OFF)
if(LIB_TORCH)
    add_definitions(-DLIB_TORCH)
    SET(LIB_TORCH_DIR /home/heaven7/heaven7/libs/libtorch-1.8.0_cu111/libtorch)
    include(${LIB_TORCH_DIR}/share/cmake/Torch/TorchConfig.cmake)
endif(LIB_TORCH)
#cuda
find_package(CUDA REQUIRED)
message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)
include_directories(${CUDA_INCLUDE_DIRS})
#nvcc flags 'https://blog.csdn.net/panda1234lee/article/details/110091930'
set(CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80;-G;-g)

message(STATUS "    libraries: ${LIBS}")

#opencv
#SET(OPENCV /home/heaven7/heaven7/libs/build-opencv-3.4.5-Desktop_Qt_5_14_2_GCC_64bit-Release)
SET(OPENCV /home/heaven7/heaven7/libs/opencv-3.4.5/.build_out/share/OpenCV)
#SET(OPENCV /home/heaven7/heaven7/libs/build-opencv-4.5.4-Desktop_Qt_5_14_2_GCC_64bit-Release)

include(${OPENCV}/OpenCVConfig.cmake)
#find_package(OpenCV 3 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


## find source and header to build
# GLOB_RECURSE/GLOB
file(GLOB CURRENT_HEADERS  cuda_opencv/*.h cuda_opencv/*.hpp cuda_opencv/*.cuh)
file(GLOB CURRENT_SOURCES  cuda_opencv/*.cpp cuda_opencv/*.cu)
source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

CUDA_ADD_EXECUTABLE(simple1
   ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_link_libraries(simple1  ${OpenCV_LIBS} ${LIBS} ${CUDA_LIBRARIES} ${RTLIBS})
if(LIB_TORCH)
    target_link_libraries(simple1 ${TORCH_LIBRARIES})
endif(LIB_TORCH)

## avx study
#add_compile_definitions(-DNDEBUG)
if(NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
endif()
file(GLOB AVX_HEADERS  avx/*.h avx/*.hpp avx/*.cuh)
file(GLOB AVX_SRCS  avx/*.cpp avx/*.cu)
set_property(SOURCE avx/study_avx.cpp APPEND_STRING PROPERTY COMPILE_FLAGS
    "-march=knl -mavx -mavx2 -mavx512f -mavx512dq -mavx512vl ")
set_property(SOURCE avx/avx_main.cpp APPEND_STRING PROPERTY COMPILE_FLAGS
    "-march=native -mavx -mavx2 -mavx512f -mavx512dq -mavx512vl ")

source_group("Include" FILES ${AVX_HEADERS})
source_group("Source" FILES ${AVX_SRCS})
add_executable(study_avx
    ${AVX_HEADERS}
    ${AVX_SRCS}
    )

### cuda-extern
#set(CUDA_SEPARABLE_COMPILATION ON)
#add_definitions(-DUSE_GPU)
#CUDA_COMPILE(EXT_B_0 cuda_extern/b.cu cuda_extern/a.cu "--gpu-architecture=sm_80 --device-c")
#message(STATUS "EXT_B_0: ${EXT_B_0}")

# not work why?
#cuda_add_executable(
#    stu_cuda_dc
#    cuda_extern/b.h
#    ${EXT_B_0}
#    )
# target_link_libraries(stu_cuda_dc ${CUDA_LIBRARIES} cudart cuda)
#
#include(cmake/test.cmake)
include(cmake/cuda_helps.cmake)
set(CUDA_SAMPLE_DIR /usr/local/cuda-11.1/samples)
set(CUDA_SIMPLE_DIR ${CUDA_SAMPLE_DIR}/0_Simple)
include_directories(${CUDA_SAMPLE_DIR}/common/inc)
#
file(GLOB SRC_ASYNC_API ${CUDA_SAMPLE_DIR}/0_Simple/asyncAPI/*.cu)
cuda_add_demo(exe_name asyncApi exe_files ${SRC_ASYNC_API})
cuda_add_demo(exe_name simpleTexture exe_files ${CUDA_SIMPLE_DIR}/simpleTexture/simpleTexture.cu)

