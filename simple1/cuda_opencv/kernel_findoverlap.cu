
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define findFirstNeqPos(a, start, len, key, out) \
{\
    int high = start + len;\
    int low = start - 1;\
    int guess = -1;\
    while(high - low > 1) {\
        guess = (high + low) / 2;\
        if (a[guess] != key) {\
            low = guess;\
        } else {\
            high = guess;\
        }\
    }\
    out = low;\
}

__inline__ __device__ int binarySearch(int* a, int start, int len, int key){
    int high = start + len;
    int low = start - 1;

    while(high - low > 1) {
        int guess = (high + low) / 2;
        if (a[guess] < key) {
            low = guess;
        } else {
            high = guess;
        }
    }

    if (high == start + len) {
        return ~(start + len);
    } else if (a[high] == key) {
        //may have the same element. we only want the first element index.
        if(high > start){
            int ret;
            findFirstNeqPos(a, start, high - start, key, ret);
            return ret + 1;
        }
        return high;
    } else {
        return ~high;
    }
}
#define HMIN(a, b) (a < b ? a : b)
#define HMAX(a, b) (a > b ? a : b)
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}

__global__ void rFindOverlaps_kernal(int outLen, int* l1,
                                     int* column,
                                     int s1_start, int s1_end,
                                     int* starts, int* ends){
    //1d
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;

    //3d
//    int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
//    int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
//    int ix = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;

    //2d
//    int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
//    int ix = threadIdx.x + blockDim.x*blockId_2D;

    if(ix < outLen){
        int j = column[ix];
        l1[ix] = HMIN(s1_end, ends[j]) - HMAX(s1_start, starts[j]);
    }
}

__global__ void rFindOverlaps_kernal2(int outLen, int* l1,
                                     int* column,
                                     int s1_start, int s1_end,
                                     int* starts, int* ends){
    //2d
    int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
    int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
    int ix = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;

    if(ix < outLen){
        int j = column[ix];
        l1[ix] = HMIN(s1_end, ends[j]) - HMAX(s1_start, starts[j]);
    }
}

//grid -> block -> thread
//blockIdx -> 所在 grid 位置
//threaIdx -> 所在 block 位置
//blockDim -> 表示线程块的大小。

// column: array, indicate all index of _starts/_ends
// _ls: array, if element >=0 means overlap.

///https://zhuanlan.zhihu.com/p/490239617: cuda二维数组
extern "C" void cuda_processOverlap(int* _ls,
                                    int* _columns, int colLen,
                                    int* _starts, int* _ends, int rc,
                                    int s1_start, int s1_end,
                                    int grid_block[4]){
    //clock_t start, end;
    static int _wrapSize = 0;
    if(_wrapSize == 0){
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if(deviceCount == 0){
            printf("no gpu.\n");
            return;
        }
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        _wrapSize = deviceProp.warpSize;
    }

    cudaError_t res;
    int* ls;
    int* columns;
    int* starts;
    int* ends;
    res = cudaMalloc((void**)&columns, colLen * sizeof(int)); CHECK(res)
    res = cudaMalloc((void**)&starts, rc * sizeof(int)); CHECK(res)
    res = cudaMalloc((void**)&ends, rc * sizeof(int)); CHECK(res)

    res = cudaMalloc((void**)&ls, colLen * sizeof(int)); CHECK(res)
    //copy
    res = cudaMemcpy((void*)(columns), (void*)(_columns), colLen * sizeof(int),
                     cudaMemcpyHostToDevice);CHECK(res)
    res = cudaMemcpy((void*)(starts), (void*)(_starts), rc * sizeof(int),
                     cudaMemcpyHostToDevice);CHECK(res)
    res = cudaMemcpy((void*)(ends), (void*)(_ends), rc * sizeof(int),
                             cudaMemcpyHostToDevice);CHECK(res)
    //kernal
   // dim3 s1; s1.x = 16; s1.y = 16; s1.z = 1;
   // dim3 s2; s2.x = colLen / 256; s2.y = 1; s2.z = 1;
            //<< <s1, s2 >> >
    if(grid_block[0] > 0){
        dim3 grid; grid.x = grid_block[0]; grid.z = 1;
        grid.y = grid_block[1] > 0 ? grid_block[1] : 1;
        //block
        if(grid_block[2] > 0){
            dim3 block; block.x = grid_block[2]; block.z = 1;
            block.y = grid_block[3] > 0 ? grid_block[3] : 1;
            rFindOverlaps_kernal2 << <grid, block >> >(colLen, ls, columns, s1_start, s1_end, starts, ends);
        }else{
            rFindOverlaps_kernal << <grid, 1 >> >(colLen, ls, columns, s1_start, s1_end, starts, ends);
        }
    }else{
        dim3 block; block.x = _wrapSize;
        if(colLen <= _wrapSize * _wrapSize){
            block.y = (colLen + block.x - 1) / block.x;
            rFindOverlaps_kernal << <block, 1 >> >(colLen, ls, columns, s1_start, s1_end, starts, ends);
        }else{
            block.y = _wrapSize;
            int n = (int)sqrt(colLen) + 1;
            dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y);
            rFindOverlaps_kernal2 << <grid, block >> >(colLen, ls, columns, s1_start, s1_end, starts, ends);
        }
    }
    cudaMemcpy(_ls, ls, colLen*sizeof(int), cudaMemcpyDeviceToHost);
    //free
    cudaFree(ls);
    cudaFree(columns);
    cudaFree(starts);
    cudaFree(ends);
}
