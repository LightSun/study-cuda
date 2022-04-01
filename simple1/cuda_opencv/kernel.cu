#include <opencv2/opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel_channel_1(uchar* srcData, uchar* dstData, int rows, int cols)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < rows && iy < cols)
    {
        *(dstData + ix + iy * rows) = *(srcData + rows - 1 - ix + (cols - 1 - iy) * rows);
    }
}

__global__ void kernel_channel_3(uchar3* srcData, uchar3* dstData, int rows, int cols)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < rows && iy < cols)
    {
        *(dstData + ix + iy * rows) = *(srcData + rows - 1 - ix + (cols - 1 - iy) * rows);
    }
}

static int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

extern "C"
void gpuMirrorImg(const cv::Mat& src, cv::Mat& dst)
{
    int rowNumber = src.rows;
    int colNumber = src.cols;

    dim3 threads(16, 16);
    dim3 grid(iDivUp(rowNumber + 15, threads.x), iDivUp(colNumber + 15, threads.y));

    size_t memSize = sizeof(uchar3) * rowNumber * colNumber;

    switch (src.channels())
    {
    case 1:
        uchar* uSrcData;
        uchar* uDstData;
        cudaMalloc((void**)&uSrcData, sizeof(uchar) * rowNumber * colNumber);
        cudaMalloc((void**)&uDstData, sizeof(uchar) * rowNumber * colNumber);

        cudaMemcpy(uSrcData, src.data, sizeof(uchar) * rowNumber * colNumber, cudaMemcpyHostToDevice);
        cudaMemset(uDstData, 0, sizeof(uchar) * rowNumber * colNumber);

        kernel_channel_1 <<<grid, threads >>>(uSrcData, uDstData, rowNumber, colNumber);

        cudaMemcpy(dst.data, uDstData, sizeof(uchar) * rowNumber * colNumber, cudaMemcpyDeviceToHost);

        // 释放空间
        cudaFree(uSrcData);
        cudaFree(uDstData);

    case 3:
        uchar3* vSrcData;
        uchar3* vDstData;
        cudaMalloc((void**)&vSrcData, memSize);
        cudaMalloc((void**)&vDstData, memSize);

        cudaMemcpy(vSrcData, src.data, memSize, cudaMemcpyHostToDevice);
        cudaMemset(vDstData, 0, memSize);

        kernel_channel_3 <<<grid, threads >>>(vSrcData, vDstData, rowNumber, colNumber);

        cudaMemcpy(dst.data, vDstData, memSize, cudaMemcpyDeviceToHost);

        //释放空间
        cudaFree(vSrcData);
        cudaFree(vDstData);

    default:
        break;
    }
}


extern "C"
void cpuMirrorImg(const cv::Mat& src, cv::Mat& dst)
{
    int rowNumber = src.rows;
    int colNumber = src.cols;

    switch (src.channels())
    {
    case 1:
        const uchar* uSrcData;
        uchar* uDstData;
        for (int i = 0; i < rowNumber; i++)
        {
            uSrcData = src.ptr<uchar>(i);
            uDstData = dst.ptr<uchar>(i);
            for (int j = 0; j < colNumber; j++)
            {
                *(uDstData + j) = *(uSrcData + colNumber - 1 - j);
            }
        }

    case 3:
        const cv::Vec3b* vSrcData;
        cv::Vec3b* vDstData;
        for (int i = 0; i < rowNumber; i++)
        {
            vSrcData = src.ptr<cv::Vec3b>(i);
            vDstData = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < colNumber; j++)
            {
                *(vDstData + j) = *(vSrcData + colNumber - 1 - j);
            }
        }
    default:
        break;
    }
}
