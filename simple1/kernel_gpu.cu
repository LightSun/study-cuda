#ifndef _KERNEL_GPU_CU_
#define _KERNEL_GPU_CU_

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;
using namespace cv::cuda;

__global__ void swap_rb_kernel(const PtrStepSz<uchar3> src, PtrStep<uchar3> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < src.cols && y < src.rows)
    {
        uchar3 v = src(y, x); // Reads pixel in GPU memory. Valid! We are on GPU!
        dst(y, x) = make_uchar3(v.z, v.y, v.x);
    }
}

void swap_rb_caller(const PtrStepSz<uchar3>& src, PtrStep<uchar3> dst, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    swap_rb_kernel <<<grid, block, 0, stream >>>(src, dst);
    if (stream == 0)
        cudaDeviceSynchronize();
}



#endif
