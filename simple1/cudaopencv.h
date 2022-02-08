#ifndef CUDAOPENCV_H
#define CUDAOPENCV_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaOpencv
{
public:
    CudaOpencv();

    /** std cv::mat with cpu/gpu */
    static void test_mat();
    static void test_iplimage();
    static void test_gpu_mat();
};

#endif // CUDAOPENCV_H
