#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "cudaopencv.h"
#include "StudyOpenMp.h"
#include "CudaDemo.h"

using namespace std;


static void printCudaInfo();

static void testCvResize();

//from: 'https://blog.csdn.net/xx116213/article/details/50704335'
int main()
{
    testCvResize();
   // cout << "Hello World!" << endl;
    printCudaInfo();
   // CudaOpencv::test1();
   // CudaOpencv::test_iplimage();
   // CudaOpencv::test_gpu_mat();
    //CudaOpencv::test_cu_opencv();
    //cv::namedWindow()
#ifdef LIB_TORCH
    extern int main_libtorch();
    main_libtorch();
#endif
    StudyOpenMp::test1();
    StudyOpenMp::test2();
    StudyOpenMp::test3();
    StudyOpenMp::test4();
    //StudyOpenMp::test5();

    CudaDemo::testTotal();
    return 0;
}


static void printCudaInfo(){
    printf("cuda deviceCount = %d\n", cv::cuda::getCudaEnabledDeviceCount());
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int dev;
    for (dev = 0; dev < deviceCount; dev++)
    {
        int driver_version(0), runtime_version(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0)
            if (deviceProp.minor = 9999 && deviceProp.major == 9999)
                printf("\n");
        printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driver_version);
        printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
        printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
        printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
        printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
        printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
        printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
        printf("Warp size:                                      %d\n", deviceProp.warpSize);
        printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
        printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
        printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
    }
}
void testCvResize(){
    auto mat = cv::imread("/home/heaven7/heaven7/study/github/mine/"
               "tensorrt-pri/trt_py/src/test_cv11.png");
    std::string path = "/home/heaven7/heaven7/work/out/kun_raw/img_11.png";
    cv::Mat src_mat = cv::imread(path);
    cv::Mat src_mat2;
    cv::resize(src_mat, src_mat2, cv::Size(1000, 1000), 0, 0, cv::INTER_LINEAR);

    cv::imshow("testCvResize", cv::abs(src_mat2 - mat));
    cv::waitKey(0);
}
