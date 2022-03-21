#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define MATRIX_SIZE 1000
#define BLOCK_SIZE 16

int DevicedChoosed = 0;
typedef struct Error {
    float max;
    float average;
}Error;

static void matMultCPU(const float* a, const float* b, float* c, int n);
static void genMat(float* arr, int n);
static Error accuracyCheck(const float* a, const float* b, int n);

//GPU并行计算矩阵乘法
__global__ void matMultCUDAKernel1(const float* a, const float* b, float* c, int n)
{
    //计算这个 thread 应该计算的 row 和 col
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    int i;
    //计算矩阵乘法 Kahan’s Summation Formula
    if (row < n && col < n)
    {
        float t = 0;
        float y = 0;
        for (i = 0; i < n; i++)
        {
            float r;

            y -= a[row * n + i] * b[i * n + col];
            r = t - y;
            y = (r - t) + y;
            t = r;
        }
        c[row * n + col] = t;
    }
}


extern "C" void test_performance0(){
    //定义矩阵
    float* a, * b, * c, * d;
    int n = MATRIX_SIZE;

    //分配host内存
    cudaMallocHost((void**)&a, sizeof(float) * n * n);
    cudaMallocHost((void**)&b, sizeof(float) * n * n);
    cudaMallocHost((void**)&c, sizeof(float) * n * n);
    d = (float*)malloc(sizeof(float) * n * n);

    genMat(a, n);
    genMat(b, n);

    float* cuda_a, * cuda_b, * cuda_c;
    clock_t start, stop;
    //分配GPU上的内存
    cudaMalloc((void**)&cuda_a, sizeof(float) * n * n);
    cudaMalloc((void**)&cuda_b, sizeof(float) * n * n);
    cudaMalloc((void**)&cuda_c, sizeof(float) * n * n);

    //拷贝数据至GPU内存
    cudaMemcpy(cuda_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    start = clock();
    //调用核函数计算
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    matMultCUDAKernel1 << <gridSize, blockSize >> > (cuda_a, cuda_b, cuda_c, n);

    //计算结果复制回主存，隐式调用同步函数
    cudaMemcpy(c, cuda_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    stop = clock();
    //释放GPU上的内存
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);
    //GPU memory
    printf("GPU memory: %e MB\n", (double)(n * n * 8) / (1024. * 1024.));
    //GPU time
    printf("GPU time: %3f ms\n", (double)(stop - start) / CLOCKS_PER_SEC * 1000.0);
    //CPU time
    start = clock();
    matMultCPU(a, b, d, n);
    stop = clock();
    printf("CPU time: %3f ms\n", (double)(stop - start) / CLOCKS_PER_SEC * 1000.0);
    //精度检测
    Error error;
    error = accuracyCheck(c, d, n);
    printf("Max error: %g Average error: %g\n", error.max, error.average);
}
//------------------------------
void matMultCPU(const float* a, const float* b, float* c, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double t = 0;
            for (int k = 0; k < n; k++)
            {
                t += (double)a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = t;
        }
    }
}
void genMat(float* arr, int n)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            arr[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
        }
    }

}

Error accuracyCheck(const float* a, const float* b, int n)
{
    Error err;
    err.max = 0;
    err.average = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (b[i * n + j] != 0)
            {
                //fabs求浮点数x的绝对值
                float delta = fabs((a[i * n + j] - b[i * n + j]) / b[i * n + j]);
                if (err.max < delta) err.max = delta;
                err.average += delta;
            }
        }
    }
    err.average = err.average / (n * n);
    return err;
}

