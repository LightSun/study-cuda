#ifndef STUDYAVX_H
#define STUDYAVX_H

#if defined(_MSC_VER)
  #include <intrin.h> // SIMD intrinsics for Windows
#else
  #include <x86intrin.h> // SIMD intrinsics for GCC
  //#include <immintrin.h>
#endif

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#ifndef uint32_t
#define uint32_t unsigned int
#endif

class StudyAVX
{
public:
    static void test1();
};

#endif // STUDYAVX_H
