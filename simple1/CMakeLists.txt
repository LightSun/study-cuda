cmake_minimum_required(VERSION 3.5)

project(simple1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#cuda
find_package(CUDA REQUIRED)
message(“CUDA_LIBRARIES:${CUDA_LIBRARIES}”)
message(“CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}”)
include_directories(${CUDA_INCLUDE_DIRS})
#nvcc
set(CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80;-G;-g)

message(STATUS "    libraries: ${LIBS}")

#opencv
#SET(OPENCV /home/heaven7/heaven7/libs/build-opencv-3.4.5-Desktop_Qt_5_14_2_GCC_64bit-Release)
SET(OPENCV /home/heaven7/heaven7/libs/opencv-3.4.5/.build_out/share/OpenCV)
include(${OPENCV}/OpenCVConfig.cmake)
#find_package(OpenCV 3 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


## find source and header to build
file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)
source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

CUDA_ADD_EXECUTABLE(simple1
   ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_link_libraries(simple1  ${OpenCV_LIBS} ${LIBS} ${CUDA_LIBRARIES} ${RTLIBS})