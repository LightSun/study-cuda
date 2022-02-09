#ifndef CUDAOPENCV_H
#define CUDAOPENCV_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/**
 //cuda::cvtColor(gpuMat, grayGpuMat, cuda::COLOR_RGB2GRAY);
 //cuda::resize
 //cv::cuda::Stream stream;
 //cv::cuda::threshold(m, m, 128, 255, cv::THRESH_BINARY, stream);

  //cuda convert to.
  cv::cuda::GpuMat m(cv::Size(1280, 1024), CV_32FC1);
  cv::cuda::Stream stream;
  m.convertTo(m, CV_8UC1, 1, 0, stream);
 */
class CudaOpencv
{
public:
    CudaOpencv();

    /** std cv::mat with cpu/gpu */
    static void test_mat();
    static void test_iplimage();
    static void test_gpu_mat();
    static void test_cu_opencv();
};

#endif // CUDAOPENCV_H
