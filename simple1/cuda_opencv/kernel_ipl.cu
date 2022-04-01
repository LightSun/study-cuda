#ifndef _KERNEL_IPL_CU_
#define _KERNEL_IPL_CU_

#include<time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define BYTE unsigned char

__global__ void InverseImg_kernel(BYTE* pImgOut, BYTE* pImgIn, int nWidth, int nHeight, int nWidthStep)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nWidth && iy < nHeight)
    {
        pImgOut[iy * nWidthStep + ix] =
            255 - pImgIn[iy * nWidthStep + ix];
    }

}

extern "C"
double cudaInverseImg(BYTE* pImgOut, BYTE* pImgIn, int nWidth, int nHeight, int nWidthStep, int nChannels)
{
    // var for timing
    clock_t start, finish;
    double  duration = 0.0;

    // cpu 计时开始
    start = clock();

    // 准备空间
    BYTE* d_pImgOut;
    BYTE* d_pImgIn;
    cudaMalloc((void**)&d_pImgOut, sizeof(BYTE) * nWidthStep * nHeight);
    cudaMalloc((void**)&d_pImgIn, sizeof(BYTE) * nWidthStep * nHeight);

    //传入数据源
    cudaMemcpy(d_pImgIn, pImgIn, sizeof(BYTE) * nWidthStep * nHeight, cudaMemcpyHostToDevice);

    cudaMemset(d_pImgOut, 0, sizeof(BYTE) * nWidthStep * nHeight);

    //ＧＰＵ处理
    dim3 ts(16, 16);
    dim3 bs((nWidth*nChannels + 15) / 16, (nHeight + 15) / 16);
    InverseImg_kernel<<< bs, ts >>>(d_pImgOut, d_pImgIn, nWidth*nChannels, nHeight, nWidthStep);


    //输出结果
    cudaMemcpy(pImgOut, d_pImgOut, sizeof(BYTE) * nWidthStep * nHeight, cudaMemcpyDeviceToHost);

    //释放空间
    cudaFree(d_pImgOut);
    cudaFree(d_pImgIn);

    //cpu 计时结束
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;

    return duration;
}

#endif
