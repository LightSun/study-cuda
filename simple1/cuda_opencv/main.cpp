#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "cudaopencv.h"
#include "StudyOpenMp.h"
#include "CudaDemo.h"
#include "Arguments.hpp"

using namespace std;

static void printCudaInfo();
static void printDeviceProp(const cudaDeviceProp& prop);
static void testCvResize();

//from: 'https://blog.csdn.net/xx116213/article/details/50704335'
int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    args.addArgument("config", "the config");
    args.addArgument("files", "the files");
    std::string val = args.get("config").as<std::string>();
    auto files = args.get("files").get();
   // ./simple1 --config efg
    //std::string str = args.usage();
    printf("config = %s\n", val.c_str());
    for(std::string abc: files){
        printf("file: %s\n", abc.c_str());
    }

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
//    StudyOpenMp::test1();
//    StudyOpenMp::test2();
//    StudyOpenMp::test3();
//    StudyOpenMp::test4();
    //StudyOpenMp::test5();

//    CudaDemo::testTotal();
//    CudaDemo::test_performance();
    CudaDemo::test_findOverlap();
    return 0;
}


static void printCudaInfo(){
    printf("cuda deviceCount = %d\n", cv::cuda::getCudaEnabledDeviceCount());
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int dev;
    for (dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printDeviceProp(deviceProp);
    }
    cudaSetDevice(0);
}
static void printDeviceProp(const cudaDeviceProp& prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %ud.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %ud.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
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
