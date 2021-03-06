/*
 Multiple N x M matrix with M x 1 matirx.
    Example of matrix multiplication equation.
                | 7 |
     | 1 2 3 |  | 8 |     | 50  |
     | 4 5 6 |  | 9 |   = | 122 |
 */

#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <stdint.h>
#include <algorithm>
#include <math.h>

#if defined(_MSC_VER)
  #include <intrin.h> // SIMD intrinsics for Windows
#else
  #include <x86intrin.h> // SIMD intrinsics for GCC
#endif

#include "common.h"

// Time.
//__rdtsc is only used for windows. cpu 指令,生成时间戳
#define rdtsc __rdtsc

#define testIterations 100000

static int preventOptimize = 0;

static float *t1;
static float *t2;
static float *out;
static float *outx8;
static float *outx16;

static const int row = 16;
static const int col = 4096;

static bool equals(const float* t1, const float* t2) {
    for (int i = 0; i < row; i++)
        if (fabs(t1[i] - t2[i]) > 0.0001) {
            return false;
        }
    return true;
}

static uint64_t scalarMultiply() {
    for (uint64_t i = 0; i < row; i++) {
        float sum = 0;
        for (uint64_t j = 0; j < col; j++)
            sum = sum + t1[i*col + j] * t2[j];
        out[i] = sum;
    }
    preventOptimize++;
    return preventOptimize;
}

static uint64_t avxMultiply() {
//    for (uint64_t i = 0; i < row; i++) {
//        __m256 sumx8 = _mm256_set1_ps(0.0);
//        for (uint64_t j = 0; j < col; j += 8) {
//            __m256 a = _mm256_loadu_ps(&(t1[i*col + j]));
//            __m256 b = _mm256_loadu_ps(&(t2[j]));
//            sumx8 = _mm256_fmadd_ps(a, b, sumx8);
//        }
//        outx8[i] = sumx8.m256_f32[0] + sumx8.m256_f32[1] +
//                  sumx8.m256_f32[2] + sumx8.m256_f32[3] +
//                  sumx8.m256_f32[4] + sumx8.m256_f32[5] +
//                  sumx8.m256_f32[6] + sumx8.m256_f32[7];
//    }
    preventOptimize++;
    return preventOptimize;
}

static uint64_t avx512Multiply() {
    for (uint64_t i = 0; i < row; i++) {
        __m512 sumx16 = _mm512_set1_ps(0.0);
        for (uint64_t j = 0; j < col; j += 16) {
            __m512 a = _mm512_loadu_ps(&(t1[i*col + j]));
            __m512 b = _mm512_loadu_ps(&(t2[j]));
            sumx16 = _mm512_fmadd_ps(a, b, sumx16);
        }
        outx16[i] = _mm512_reduce_add_ps(sumx16);
    }
    preventOptimize++;
    return preventOptimize;
}

static bool init() {
    t1 = (float *)_aligned_malloc(row * col * sizeof(float), 64);
    t2 = (float *)_aligned_malloc(col * sizeof(float), 64);

    out = new float[row];
    outx8 = new float[row];
    outx16 = new float[row];

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            t1[i*col + j] = 2.0;
        }
    }
    for (int j = 0; j < col; j++) {
        t2[j] = 3.0;
    }
    scalarMultiply();
    //avxMultiply();
    avx512Multiply();

    //return equals(out, outx8) && equals(out, outx16);
    return equals(out, outx16);
}

static bool cleanup() {
    _aligned_free(t1);
    _aligned_free(t2);

    delete[] out;
    delete[] outx8;
    delete[] outx16;

    return true;
};


// the more to study: 'https://github.com/intel/
// Developer-Tools-Runtimes-Blogs/blob/master/AVX512_Blog/MandelbrotImage.cpp'
int main(void) {
    static uint64_t start, end;
    double dtAvx, dtAvx512;
    uint64_t val = 0;
    if (!init()) {
        printf("Wrong implementation.\n");
        return 0;
    }
    // normal
    start = rdtsc();
    for (int i = 0; i < testIterations; ++i) {
        val = scalarMultiply();
    }
    end = rdtsc();
    dtAvx = end - start;

    // AVX2
//    start = rdtsc();
//    for (int i = 0; i < testIterations; ++i) {
//        val = avxMultiply();
//    }
//    end = rdtsc();
//    dtAvx = end - start;

    // AVX512
    start = rdtsc();
    for (int i = 0; i < testIterations; ++i) {
        val = avx512Multiply();
    }
    end = rdtsc();
    dtAvx512 = end - start;
//    if (!equals(outx16, outx8)) {
//        printf("Wrong result. \n");
//        return 0;
//    }
    printf("AVX2/AVX512 = %f", dtAvx / dtAvx512);
    //avx512 with normal. 4.517527
    cleanup();
    return preventOptimize;
}
